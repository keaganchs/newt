import math
import copy
import random
import einops

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pydantic import BaseModel

from config import Config
from common.layers import mlp
from common.trm.trm_layers import trunc_normal_init_, rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, CastedSparseEmbedding


IGNORE_LABEL_ID = -100


@dataclass
class TRMInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TRMCarry:
    inner_carry: TRMInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    
    task_emb_len: int = 16 # if non-zero, length of task embedding
    task_dim: int = 0 # previously task_emb_ndim: int = 0 # Dim of task embedding space. Set to 0 to disable
    # num_task_identifiers: int # Number of unique tasks
    vocab_size: int # Number of tokens in the model's vocabulary 

    H_cycles: int
    L_cycles: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False # use mlp on L instead of transformer


class TRMBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        
        # MLP or Attention layers
        if self.config.mlp_t:
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len, # L
                expansion=config.expansion,
            )

            # Pure MLP version
            # self.mlp_t = mlp(
            #     in_dim=self.config.seq_len, # L
            #     mlp_dims=max(self.config.num_enc_layers-1, 1)*[self.config.enc_dim],
            #     out_dim=self.config.seq_len,
            # )
        else:
            self.self_attn = Attention(
                hidden_size=self.config.hidden_size,
                head_dim=self.config.hidden_size // self.config.num_heads,
                num_heads=self.config.num_heads,
                num_key_value_heads=self.config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

        # Pure MLP version
        # self.mlp = mlp(
        #     in_dim=self.config.hidden_size,
        #     mlp_dims=max(self.config.num_enc_layers-1, 1)*[self.config.enc_dim],
        #     out_dim=self.config.hidden_size,
        # )
        self.norm_eps = self.config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm. Adding .contiguous() gives a small speedup to the matrix multiplications
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2).contiguous()
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2).contiguous()
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRMReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TRMInner(nn.Module):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.forward_dtype = torch.bfloat16
        # Get pytorch dtype from config string
        if hasattr(self.config, 'forward_dtype'):
            try:
                self.forward_dtype = getattr(torch, self.config.forward_dtype)
            except AttributeError:
                raise ValueError(f"Invalid torch dtype: {self.config.forward_dtype}")

        super().__init__()

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        # TODO: handle continuous inputs properly: currently checks if input is floating point and uses a linear layer
        self.embed_continuous = CastedLinear(1, self.config.hidden_size, bias=False)
        with torch.no_grad():
            trunc_normal_init_(self.embed_continuous.weight, std=embed_init_std)

        # TODO: Accept device arg
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.latent_dim, bias=False).to(device="cuda")
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True).to(device="cuda")
        
        if self.config.task_dim > 0:
            # Zero init task embeddings
            # TODO: add device arg
            num_tasks = len(self.config.task_embeddings) if self.config.task_embeddings is not None else 1
            # Newt trains on a flattened batch of sequences with shape (batch_size * horizon)
            effective_batch_size = self.config.batch_size * self.config.horizon
            self.task_emb = CastedSparseEmbedding(num_tasks, self.config.task_dim,
                                                    batch_size=effective_batch_size, init_std=0, cast_to=self.forward_dtype).to(device="cuda")
            
            # Initialize with task_embeddings from config if available
            if self.config.task_embeddings is not None:
                try:
                    pretrained_weights = torch.tensor(self.config.task_embeddings, dtype=torch.float32)
                    if pretrained_weights.shape == self.task_emb.weights.shape:
                        with torch.no_grad():
                            self.task_emb.weights.copy_(pretrained_weights)
                            print(f"Initialized TRM task embeddings from config with shape {pretrained_weights.shape}")
                except Exception as e:
                    print(f"Failed to initialize TRM task embeddings from config: {e}")

        # LM Blocks
        # TODO: add device arg
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta).to(device="cuda")
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype).to(device="cuda")
        else:
            pass

        # Reasoning Layers
        self.L_level = TRMReasoningModule(layers=[TRMBlock(self.config).to(device="cuda") for _i in range(self.config.L_layers)])

        # Initial states
        # TODO: add device arg
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1).to(device="cuda"), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1).to(device="cuda"), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, task_embedding: torch.Tensor):
        # Token embedding
        # TODO: update code to handle continuous inputs (add discretization flag, int casting only when enabled; check tokeniztion/vocab size etc.)
        if input.is_floating_point():
            embedding = self.embed_continuous(input.unsqueeze(-1))
        else:
            embedding = self.embed_tokens(input.to(torch.int32))

        # Task embeddings
        if self.config.task_dim > 0:
            task_embedding = self.task_emb(task_embedding)
            
            pad_count = self.config.task_emb_len * self.config.hidden_size - task_embedding.shape[-1]
            if pad_count > 0:
                task_embedding = F.pad(task_embedding, (0, pad_count))

            embedding = torch.cat((task_embedding.view(-1, self.config.task_emb_len, self.config.hidden_size), embedding), dim=-2)
        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None) -> TRMInnerCarry:
        return TRMInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry):
        return TRMInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["task_embedding"]).to(device=carry.z_H.device)

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        
        # H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.config.H_cycles-1): # H step
                for _ in range(self.config.L_cycles): # L step
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _ in range(self.config.L_cycles): # L step
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        # TODO: add device arg
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.config.task_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first task_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRM(nn.Module):
    """Tiny Recursion Model (TRM) with Adaptive Computation Time (ACT)"""

    def __init__(self, config: Config):
        self.config = config

        # Calculate task embedding sequence length (number of tokens needed to represent task_dim)
        if self.config.task_dim > 0:
             self.config.task_emb_len = -(self.config.task_dim // -self.config.hidden_size) # ceil div
        else:
             self.config.task_emb_len = 0

        # Calculate total sequence length
        if self.config.obs == 'state':
            # Note: hidden_size is kept from config (e.g., 256) to maintain model size. 
            self.config.seq_len = self.config.obs_shape['state'][0] + self.config.task_emb_len
        elif self.config.obs == 'rgb':
            self.config.seq_len = self.config.obs_shape['state'][0] + self.config.obs_shape['rgb'][0] + self.config.task_emb_len
        else:
            raise NotImplementedError(f"Unexpected observation type: {self.config.obs}")
        
        super().__init__()
        # TODO: Accept device arg
        self.inner = TRMInner(self.config).to(torch.device('cuda'))
        # self.out_proj = nn.Linear(config.hidden_size, config.latent_dim + config.task_dim)

    @property
    def task_emb(self):
        return self.inner.task_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        # TODO: log halted
        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {}

        for k, v in carry.current_data.items():
            if batch[k].shape[0] != carry.halted.shape[0]:
                raise ValueError(f"Batch dimension mismatch for key '{k}'. Expected {carry.halted.shape[0]} (based on halted), but got {batch[k].shape[0]}.")
            new_current_data[k] = torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v)

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        # Cast logit dtype to float32 for gym compatibility
        logits = logits.to(torch.float32)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > 0)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)


        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
