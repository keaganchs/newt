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
from common.layers import mlp, SimNorm
from common.trm.trm_layers import trunc_normal_init_, rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, CastedSparseEmbedding

"""
Tiny Recursive Model (TRM), copied from the original implementation and modified
"""


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


class TRMBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.tokenize = getattr(self.config, 'trm_tokenize', True)
        
        # Token mixing layers (disabled in flat mode)
        if self.tokenize:
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
        # Channel mixing MLP (operates on hidden_size in tokenized mode, flat_dim in flat mode)
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
        # Token mixing (disabled in flat mode)
        if self.tokenize:
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
        # Channel mixing
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
        self.tokenize = getattr(self.config, 'trm_tokenize', True)

        # I/O
        self.embed_scale = 1.0 # math.sqrt(self.config.hidden_size)
        embed_init_std = 0.02 # 1.0 / self.embed_scale

        # State observation: if tokenized, project each patch of num_state_obs_per_token scalars to hidden_size
        # self.embed_tokens = CastedEmbedding(self.config.num_tasks, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        if self.tokenize:
            self.embed_state_obs = CastedLinear(self.config.num_state_obs_per_token, self.config.hidden_size, bias=False)
            with torch.no_grad():
                trunc_normal_init_(self.embed_state_obs.weight, std=embed_init_std)

        # Task embedding projection (tokenized mode only)
        # If task_dim == hidden_size: use the CLIP token directly as 1 token (no projection)
        # If task_dim < hidden_size: project up to hidden_size -> 1 token
        # If task_dim > hidden_size: project to num_task_tokens * hidden_size -> multiple tokens
        if self.tokenize and self.config.task_dim > 0 and self.config.task_dim != self.config.hidden_size:
            if self.config.task_dim < self.config.hidden_size:
                # Project up: (task_dim,) -> (hidden_size,) = 1 token
                self.task_proj = CastedLinear(self.config.task_dim, self.config.hidden_size, bias=False).to(device="cuda")
            else:
                # Project to multiple tokens: (task_dim,) -> (num_task_tokens * hidden_size,)
                self.task_proj = CastedLinear(self.config.task_dim, self.config.num_task_tokens * self.config.hidden_size, bias=False).to(device="cuda")
            with torch.no_grad():
                trunc_normal_init_(self.task_proj.weight, std=embed_init_std)

        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.latent_dim, bias=False).to(device="cuda")
        with torch.no_grad():
            trunc_normal_init_(self.lm_head.weight, std=0.02)  # Match Newt's default nn.Linear init
        self.lm_head_norm = SimNorm(self.config)  # SimNorm to match baseline encoder output distribution
        if self.config.use_trm_hidden_state_simnorm:
            self.embed_norm = SimNorm(self.config)  # SimNorm on input to help match the output distribution
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True).to(device="cuda") # TODO: check q_head needs 2 outputs or just 1 for halt logit
        
        # Task embedding: frozen CLIP embeddings stored as a buffer for state_dict compatibility
        if self.config.task_dim > 0:
            num_tasks = len(self.config.task_embeddings) if self.config.task_embeddings is not None else 1
            
            # Initialize with task_embeddings from config if available
            if self.config.task_embeddings is not None:
                _task_emb = torch.tensor(self.config.task_embeddings, dtype=self.forward_dtype)
            else: # Default to truncated normal init
                _task_emb = trunc_normal_init_(torch.empty((num_tasks, self.config.task_dim), dtype=self.forward_dtype), std=embed_init_std)
            self.task_emb_init = nn.Buffer(_task_emb.to(device="cuda"), persistent=True)

        # Position encodings (tokenized mode only)
        if self.tokenize:
            if self.config.pos_encodings == "rope":
                self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                                  max_position_embeddings=self.config.seq_len,
                                                  base=self.config.rope_theta).to(device="cuda")
            elif self.config.pos_encodings == "learned":
                self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype).to(device="cuda")

        # Reasoning Layers
        self.L_level = TRMReasoningModule(layers=[TRMBlock(self.config).to(device="cuda") for _ in range(self.config.L_layers)])

        # Initial carry states (zero-init: carry is recreated fresh every encode() call,
        # so non-zero init just adds fixed noise that overwhelms the input signal)
        # TODO: test trunc_normal_init_ with updated std 
        self.H_init = nn.Buffer(torch.zeros(self.config.hidden_size, dtype=self.forward_dtype, device="cuda"), persistent=True)
        self.L_init = nn.Buffer(torch.zeros(self.config.hidden_size, dtype=self.forward_dtype, device="cuda"), persistent=True)

        # [CLS] token: learnable aggregation token prepended to position 0 (tokenized mode only)
        if self.tokenize and self.config.pooling_strategy == "cls":
            self.cls_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=embed_init_std).to(device="cuda"), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, task_embedding: torch.Tensor):
        """Process input into token embeddings (tokenized) or a flat vector (flat mode)."""
        batch_size = input.shape[0]

        if not input.is_floating_point():
            raise NotImplementedError("TRM currently only supports continuous inputs.")

        # Flat mode: concatenate raw obs and task embeddings, recurse, then project with lm_head
        if not self.tokenize:
            parts = [input]  # (batch_size, obs_dim)
            if self.config.task_dim > 0:
                task_vec = self.task_emb_init[task_embedding.long()]  # (batch_size, task_dim)
                parts.append(task_vec)
            out = torch.cat(parts, dim=-1).to(self.forward_dtype)  # (batch_size, flat_dim)
            if self.config.use_trm_hidden_state_simnorm:
                out = self.embed_norm(out) - (1.0 / self.config.simnorm_dim)
            return out

        # Tokenized mode: patch obs (and task embedding) into tokens
        # State observation patching
        # input: (batch_size, obs_dim)
        obs = input
        # Pad obs so it's divisible by num_state_obs_per_token
        if self.config.obs_pad_len > 0:
            obs = F.pad(obs, (0, self.config.obs_pad_len))  # (batch_size, obs_dim + pad)
        # Reshape into patches: -> (batch_size, num_state_tokens, num_state_obs_per_token)
        obs_patches = obs.view(batch_size, self.config.num_state_tokens, self.config.num_state_obs_per_token)
        # Project each patch: -> (batch_size, num_state_tokens, hidden_size)
        obs_embedding = self.embed_state_obs(obs_patches)

        # Task embedding
        # task_embedding is an integer task index; look up the precomputed CLIP embedding and convert to token(s)
        tokens = [obs_embedding]  # list of (batch_size, num_tokens_i, hidden_size)
        if self.config.task_dim > 0:
            # Look up precomputed CLIP embeddings by task index
            task_vec = self.task_emb_init[task_embedding.long()]  # (batch_size, task_dim)

            if self.config.task_dim == self.config.hidden_size:
                # Exact match: use the CLIP token directly as a single token
                task_tokens = task_vec.unsqueeze(1)  # (batch_size, 1, hidden_size)
            elif self.config.task_dim < self.config.hidden_size:
                # Project up: (batch_size, task_dim) -> (batch_size, hidden_size) = 1 token
                task_tokens = self.task_proj(task_vec).unsqueeze(1)  # (batch_size, 1, hidden_size)
            else:
                # Project to multiple tokens: (batch_size, task_dim) -> (batch_size, num_task_tokens * hidden_size)
                task_tokens = self.task_proj(task_vec).view(batch_size, self.config.num_task_tokens, self.config.hidden_size)

            tokens.insert(0, task_tokens)  # task tokens before obs tokens

        # RGB embedding
        if self.config.obs == 'rgb':
            raise NotImplementedError("RGB observations are not yet implemented with the TRM architecture.")

        # Concatenate all content tokens: (batch_size, num_task_tokens + num_state_tokens, hidden_size)
        content = torch.cat(tokens, dim=1)

        # Position embedding (if learned)
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            content = 0.707106781 * (content + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        out = (content * self.embed_scale).to(self.forward_dtype)

        if self.config.use_trm_hidden_state_simnorm:
            out = self.embed_norm(out) - (1.0 / self.config.simnorm_dim)

        # Prepend [CLS] token at position 0 (only when using CLS pooling)
        if self.config.pooling_strategy == "cls":
            cls_token = self.cls_init.unsqueeze(0).expand(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
            out = torch.cat([cls_token, out], dim=1)  # (batch_size, seq_len, hidden_size)

        return out

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None) -> TRMInnerCarry:
        if not self.tokenize:
            # Flat mode: carry is (batch, hidden_size) (note hidden_size == flat_dim is set automatically)
            return TRMInnerCarry(
                z_H=torch.empty(batch_size, self.config.hidden_size, dtype=self.forward_dtype, device=device),
                z_L=torch.empty(batch_size, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            )
        # Tokenized mode: carry is (batch, seq_len, hidden_size)
        return TRMInnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TRMInnerCarry):
        if not self.tokenize:
            # Flat mode: (batch, hidden_size), broadcast flag as (batch, 1)
            return TRMInnerCarry(
                z_H=torch.where(reset_flag.view(-1, 1), self.H_init, carry.z_H),
                z_L=torch.where(reset_flag.view(-1, 1), self.L_init, carry.z_L),
            )
        # Tokenized mode: (batch, seq_len, hidden_size), broadcast flag as (batch, 1, 1)
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
                z_H_inject = z_H + input_embeddings  # Precompute once per H-cycle
                for _ in range(self.config.L_cycles): # L step
                    z_L = self.L_level(z_L, z_H_inject, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        z_H_inject = z_H + input_embeddings
        for _ in range(self.config.L_cycles): # L step
            z_L = self.L_level(z_L, z_H_inject, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        if not self.tokenize:
            # No pooling needed
            pooled = z_H
        elif self.config.pooling_strategy == "cls":
            # [CLS] token at position 0
            pooled = z_H[:, 0]  # (batch_size, hidden_size)
        else:
            # Mean pooling over all sequence positions
            pooled = z_H.mean(dim=1)  # (batch_size, hidden_size)
        output = self.lm_head_norm(self.lm_head(pooled).to(torch.float32))  # (batch_size, latent_dim), SimNorm-normalized
        q_logits = self.q_head(pooled).to(torch.float32)  # (batch_size, 2)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRM(nn.Module):
    """Tiny Recursion Model (TRM) with Adaptive Computation Time (ACT)"""

    def __init__(self, config: Config):
        self.config = config

        if not getattr(self.config, 'trm_tokenize', True):
            # Flat/vector mode
            # Obs and task are concatenated into a single vector.
            # hidden_size is set to flat_dim (obs_dim + task_dim)
            # No (token) mixing
            # Carry shape: (batch, flat_dim) instead of (batch, seq_len, hidden_size).
            obs_dim = self.config.obs_shape['state'][0]
            flat_dim = obs_dim + (self.config.task_dim if self.config.task_dim > 0 else 0)
            self.config.flat_dim = flat_dim
            self.config.hidden_size = flat_dim  # SwiGLU channel MLP size = flat vector length
            self.config.seq_len = 1             # not used in flat mode
            self.config.num_task_tokens = 0
            self.config.num_state_tokens = 0
            self.config.obs_pad_len = 0
        else:
            # Tokenized mode
            # Calculate length of task embedding chunks 
            if self.config.task_dim > 0:
                 self.config.num_task_tokens = -(self.config.task_dim // -self.config.hidden_size) # ceil div
            else:
                 self.config.num_task_tokens = 0

            # Calculate obs patch sizes. One token will share a number of observations (e.g. 16) to reduce sequence length
            obs_dim = self.config.obs_shape['state'][0]
            if self.config.num_state_obs_per_token > 1:
                # ceil(obs_dim / num_state_obs_per_token)
                self.config.num_state_tokens = -(obs_dim // -self.config.num_state_obs_per_token)
                # Pad size: how many zeros to append so obs_dim is divisible by num_state_obs_per_token
                self.config.obs_pad_len = self.config.num_state_tokens * self.config.num_state_obs_per_token - obs_dim
            else: 
                # Project each scalar in the observation vector to its own token
                self.config.num_state_tokens = obs_dim
                self.config.obs_pad_len = 0

            if self.config.obs == 'rgb':
                raise NotImplementedError("RGB Observations are not yet implemented with the TRM architecture.")
                # TODO: check vision encoder token size(s). Just 1 512-dim token?
                self.config.num_rgb_tokens = -(self.config.obs_shape['rgb'][0] // -self.config.hidden_size)

            # Calculate total sequence length (+1 for [CLS] token at position 0)
            if self.config.obs == 'state':
                self.config.seq_len = (1 if self.config.pooling_strategy == "cls" else 0) + self.config.num_state_tokens + self.config.num_task_tokens
            elif self.config.obs == 'rgb':
                self.config.seq_len = (1 if self.config.pooling_strategy == "cls" else 0) + self.config.num_state_tokens + self.config.num_task_tokens + self.config.num_rgb_tokens # TODO: rgb may disable state observations. This is not ideal for VLA-type tasks
            else:
                raise NotImplementedError(f"Unexpected observation type: {self.config.obs}")
        
        super().__init__()
        self.inner = TRMInner(self.config).to(torch.device('cuda'))

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),  # Empty is expected, it will be reset in first pass as all sequences are halted.
            
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
