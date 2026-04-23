from typing import Tuple, List, Dict, Optional, Callable, Any
from dataclasses import dataclass

import torch
from torch import nn

from config import Config
from common.layers import mlp, SimNorm
from common.trm.trm_layers import trunc_normal_init_, rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear

"""
Tiny Recursive Model (TRM), copied from the original implementation and modified
"""


IGNORE_LABEL_ID = -100


def _resolve_scan_impl() -> Optional[Callable[..., Tuple[Any, Any]]]:
    if hasattr(torch, "scan"):
        return getattr(torch, "scan")
    
    from torch._higher_order_ops import scan
    return scan


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
        
        # MLP or Attention layers
        if self.config.mlp_t:
            if self.config.trm_mlp_mixer_type == "swiglu":
                self.mlp_t = SwiGLU(
                    hidden_size=self.config.seq_len, # L
                    expansion=config.expansion,
                )
            elif self.config.trm_mlp_mixer_type == "simnorm":
                self.mlp_t = mlp(
                    in_dim=self.config.seq_len, # L
                    mlp_dims=max(self.config.num_enc_layers-1, 1)*[self.config.enc_dim],
                    out_dim=self.config.seq_len,
                    act=SimNorm(self.config)
                )
            else:
                raise ValueError(f"Unsupported TRM MLP mixer type: {self.config.trm_mlp_mixer_type}")
            self._token_mixer = self._token_mixer_mlp
        else:
            self.self_attn = Attention(
                hidden_size=self.config.hidden_size,
                head_dim=self.config.hidden_size // self.config.num_heads,
                num_heads=self.config.num_heads,
                num_key_value_heads=self.config.num_heads,
                causal=False,
                use_rope=self.config.pos_encodings == "rope",
            )
            self._token_mixer = self._token_mixer_attn
        if self.config.trm_mlp_output_type == "swiglu":
            self.mlp = SwiGLU(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
        elif self.config.trm_mlp_output_type == "simnorm":
            self.mlp = mlp(
                in_dim=self.config.hidden_size,
                mlp_dims=max(self.config.num_enc_layers-1, 1)*[self.config.enc_dim],
                out_dim=self.config.hidden_size,
                act=SimNorm(self.config)
            )
        else:
            raise ValueError(f"Unsupported TRM output MLP type: {self.config.trm_mlp_output_type}")
        
        self.norm_eps = self.config.rms_norm_eps

    def _token_mixer_mlp(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        out = self.mlp_t(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states.transpose(1, 2).contiguous()

    def _token_mixer_attn(self, hidden_states: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        return rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )

    @torch.compile(fullgraph=True)
    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._token_mixer(hidden_states, cos_sin)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRMReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    @torch.compile(fullgraph=True)
    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin)
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
        self.embed_scale = 1.0 # math.sqrt(self.config.hidden_size)
        embed_init_std = 0.02 # 1.0 / self.embed_scale

        # State observation: project each patch of num_state_obs_per_token scalars to hidden_size
        # self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.embed_state_obs = CastedLinear(self.config.num_state_obs_per_token, self.config.hidden_size, bias=False)
        with torch.no_grad():
            trunc_normal_init_(self.embed_state_obs.weight, std=embed_init_std)

        # Task embedding projection: handles task_dim != hidden_size
        self._use_task_tokens = self.config.task_dim > 0
        if self._use_task_tokens and self.config.task_dim != self.config.hidden_size:
            if self.config.task_dim < self.config.hidden_size:
                self.task_proj = CastedLinear(self.config.task_dim, self.config.hidden_size, bias=False)
                self._task_tokenizer = self._task_tokens_proj_single
            else:
                self.task_proj = CastedLinear(self.config.task_dim, self.config.num_task_tokens * self.config.hidden_size, bias=False)
                self._task_tokenizer = self._task_tokens_proj_multi
            with torch.no_grad():
                trunc_normal_init_(self.task_proj.weight, std=embed_init_std)
        elif self._use_task_tokens:
            self._task_tokenizer = self._task_tokens_direct
        else:
            self._task_tokenizer = self._task_tokens_empty

        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.latent_dim, bias=False)
        with torch.no_grad():
            trunc_normal_init_(self.lm_head.weight, std=0.02)  # Match Newt's default nn.Linear init
        self.lm_head_norm = SimNorm(self.config)  # SimNorm to match baseline encoder output distribution
        if self.config.use_trm_hidden_state_simnorm:
            self.embed_norm = SimNorm(self.config)  # SimNorm on input to help match the output distribution
            self._apply_embed_norm = self._apply_embed_norm_simnorm
        else:
            self._apply_embed_norm = self._apply_embed_norm_identity
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True) # TODO: check q_head needs 2 outputs or just 1 for halt logit
        
        # Task embedding: frozen CLIP embeddings stored as a buffer for state_dict compatibility
        if self._use_task_tokens:
            num_tasks = len(self.config.task_embeddings) if self.config.task_embeddings is not None else 1
            
            # Initialize with task_embeddings from config if available
            if self.config.task_embeddings is not None:
                _task_emb = torch.tensor(self.config.task_embeddings, dtype=self.forward_dtype)
            else: # Default to truncated normal init
                _task_emb = trunc_normal_init_(torch.empty((num_tasks, self.config.task_dim), dtype=self.forward_dtype), std=embed_init_std)
            self.task_emb_init = nn.Buffer(_task_emb, persistent=True)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta)
            self._get_cos_sin = self._get_cos_sin_rope
            self._apply_positional = self._apply_positional_identity
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
            self._get_cos_sin = self._get_cos_sin_none
            self._apply_positional = self._apply_positional_learned
        else:
            self._get_cos_sin = self._get_cos_sin_none
            self._apply_positional = self._apply_positional_identity

        self._obs_pad = nn.ConstantPad1d((0, self.config.obs_pad_len), 0.0) if self.config.obs_pad_len > 0 else nn.Identity()

        # Reasoning Layers
        self.L_level = TRMReasoningModule(layers=[TRMBlock(self.config) for _ in range(self.config.L_layers)])

        # Scan loop placeholders (kept as buffers to stay on model device)
        self.l_scan_tokens = nn.Buffer(torch.arange(self.config.L_cycles, dtype=torch.int32), persistent=False)
        self.h_nograd_tokens = nn.Buffer(torch.arange(max(self.config.H_cycles - 1, 0), dtype=torch.int32), persistent=False)
        self.h_grad_tokens = nn.Buffer(torch.arange(1, dtype=torch.int32), persistent=False)
        self.has_h_nograd_scan = self.config.H_cycles > 1
        self._scan_impl = _resolve_scan_impl()
        if self._scan_impl is None:
            raise RuntimeError("TRM requires torch.scan (or torch.func.scan) for recursion loops.")

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)

        # [CLS] token: learnable aggregation token prepended to position 0
        if self.config.pooling_strategy == "cls":
            self.cls_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=embed_init_std), persistent=True)
            self._prepend_cls = self._prepend_cls_token
            self._pool = self._pool_cls
        elif self.config.pooling_strategy == "mean":
            self._prepend_cls = self._prepend_cls_identity
            self._pool = self._pool_mean
        elif self.config.pooling_strategy == "mean_obs_only":
            self._prepend_cls = self._prepend_cls_identity
            self._pool = self._pool_mean_obs_only
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.config.pooling_strategy}")

        self.obs_token_start = (1 if self.config.pooling_strategy == "cls" else 0) + self.config.num_task_tokens
        self.obs_token_end = self.obs_token_start + self.config.num_state_tokens
        self.simnorm_offset = 1.0 / self.config.simnorm_dim

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _scan(self, fn: Callable[[Any, torch.Tensor], Tuple[Any, Any]], init: Any, xs: torch.Tensor) -> Tuple[Any, Any]:
        return self._scan_impl(fn, init, xs)

    def _task_tokens_empty(self, task_embedding: torch.Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.empty(batch_size, 0, self.config.hidden_size, dtype=dtype, device=device)

    def _task_tokens_direct(self, task_embedding: torch.Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.task_emb_init[task_embedding.long()].unsqueeze(1).to(dtype=dtype, device=device)

    def _task_tokens_proj_single(self, task_embedding: torch.Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        task_vec = self.task_emb_init[task_embedding.long()]
        return self.task_proj(task_vec).unsqueeze(1).to(dtype=dtype, device=device)

    def _task_tokens_proj_multi(self, task_embedding: torch.Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        task_vec = self.task_emb_init[task_embedding.long()]
        return self.task_proj(task_vec).view(batch_size, self.config.num_task_tokens, self.config.hidden_size).to(dtype=dtype, device=device)

    def _apply_positional_identity(self, content: torch.Tensor) -> torch.Tensor:
        return content

    def _apply_positional_learned(self, content: torch.Tensor) -> torch.Tensor:
        return 0.707106781 * (content + self.embed_pos.embedding_weight.to(content.dtype))

    def _apply_embed_norm_identity(self, out: torch.Tensor) -> torch.Tensor:
        return out

    def _apply_embed_norm_simnorm(self, out: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(out) - self.simnorm_offset

    def _prepend_cls_identity(self, out: torch.Tensor, batch_size: int) -> torch.Tensor:
        return out

    def _prepend_cls_token(self, out: torch.Tensor, batch_size: int) -> torch.Tensor:
        cls_token = self.cls_init.unsqueeze(0).expand(batch_size, 1, -1).to(dtype=out.dtype, device=out.device)
        return torch.cat((cls_token, out), dim=1)

    def _pool_cls(self, z_h: torch.Tensor) -> torch.Tensor:
        return z_h[:, 0]

    def _pool_mean(self, z_h: torch.Tensor) -> torch.Tensor:
        return z_h.mean(dim=1)

    def _pool_mean_obs_only(self, z_h: torch.Tensor) -> torch.Tensor:
        return z_h[:, self.obs_token_start:self.obs_token_end].mean(dim=1)

    def _get_cos_sin_none(self) -> Optional[CosSin]:
        return None

    def _get_cos_sin_rope(self) -> Optional[CosSin]:
        return self.rotary_emb()

    def _run_l_scan(self, z_l: torch.Tensor, z_h_inject: torch.Tensor, cos_sin: Optional[CosSin]) -> torch.Tensor:
        # Nested scan + compile currently triggers autograd tracing failures in some PyTorch builds.
        # Keep outer H recursion on scan, and run the inner L recursion as a static loop.
        for _ in range(self.config.L_cycles):
            z_l = self.L_level(z_l, z_h_inject, cos_sin=cos_sin)
        return z_l

    def _input_embeddings(self, input: torch.Tensor, task_embedding: torch.Tensor):
        """Process input (observation; x) into tokens, which are then summed with the hidden states (y and z)"""
        batch_size = input.shape[0]

        # State observation patching
        # input: (batch_size, obs_dim)
        obs = self._obs_pad(input)
        # Reshape into patches: -> (batch_size, num_state_tokens, num_state_obs_per_token)
        obs_patches = obs.view(batch_size, self.config.num_state_tokens, self.config.num_state_obs_per_token)
        # Project each patch: -> (batch_size, num_state_tokens, hidden_size)
        obs_embedding = self.embed_state_obs(obs_patches)

        task_tokens = self._task_tokenizer(
            task_embedding,
            batch_size,
            dtype=obs_embedding.dtype,
            device=obs_embedding.device,
        )

        # Concatenate all content tokens: (batch_size, num_task_tokens + num_state_tokens, hidden_size)
        content = torch.cat((task_tokens, obs_embedding), dim=1)
        content = self._apply_positional(content)

        # Scale
        out = (content * self.embed_scale).to(self.forward_dtype)
        out = self._apply_embed_norm(out)
        return self._prepend_cls(out, batch_size)

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

    @torch.compile(fullgraph=True, dynamic=True)
    def forward(self, carry: TRMInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = self._get_cos_sin()

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["task_embedding"]).to(device=carry.z_H.device)

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 without grad
        if self.has_h_nograd_scan:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    z_H_inject = z_H + input_embeddings
                    z_L = self._run_l_scan(z_L, z_H_inject, cos_sin)
                    z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # Final H step with grad
        z_H_inject = z_H + input_embeddings
        z_L = self._run_l_scan(z_L, z_H_inject, cos_sin)
        z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)

        # LM Outputs
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        pooled = self._pool(z_H)
            
        output = self.lm_head_norm(self.lm_head(pooled).to(torch.float32))  # (batch_size, latent_dim), SimNorm-normalized
        q_logits = self.q_head(pooled).to(torch.float32)  # (batch_size, 2)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TRM(nn.Module):
    """Tiny Recursion Model (TRM) with Adaptive Computation Time (ACT)"""

    def __init__(self, config: Config, model_type="encoder"):
        """Initializes the TRM model
        Args:
            config: Config object containing model hyperparameters
            model_type: "encoder" or "dynamics", used to determine input/output dimensions
        """
        self.config = config
        self.model_type = model_type

        # Calculate length of task embedding chunks 
        if self.config.task_dim > 0:
             self.config.num_task_tokens = -(self.config.task_dim // -self.config.hidden_size) # ceil div
        else:
             self.config.num_task_tokens = 0

        # Calculate obs patch sizes. One token will share a number of observations (e.g. 16) to reduce sequence length
        if self.model_type == "encoder":
            obs_dim = self.config.obs_shape['state'][0]
        elif self.model_type == "dynamics":
            obs_dim = self.config.latent_dim + self.config.action_dim
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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

        # Calculate total sequence length
        if self.config.obs == 'state':
            self.config.seq_len = (1 if self.config.pooling_strategy == "cls" else 0) + self.config.num_state_tokens + self.config.num_task_tokens
        elif self.config.obs == 'rgb':
            self.config.seq_len = (1 if self.config.pooling_strategy == "cls" else 0) + self.config.num_state_tokens + self.config.num_task_tokens + self.config.num_rgb_tokens # TODO: rgb may disable state observations. This is not ideal for VLA-type tasks
        else:
            raise NotImplementedError(f"Unexpected observation type: {self.config.obs}")

        # SimNorm requires its target dimension to be divisible by simnorm_dim.
        if self.config.mlp_t and self.config.trm_mlp_mixer_type == "simnorm":
            if self.config.seq_len % self.config.simnorm_dim != 0:
                raise ValueError(
                    "Invalid TRM config for SimNorm token mixer: "
                    f"seq_len={self.config.seq_len} must be divisible by simnorm_dim={self.config.simnorm_dim}."
                )
        if self.config.trm_mlp_output_type == "simnorm":
            if self.config.hidden_size % self.config.simnorm_dim != 0:
                raise ValueError(
                    "Invalid TRM config for SimNorm output MLP: "
                    f"hidden_size={self.config.hidden_size} must be divisible by simnorm_dim={self.config.simnorm_dim}."
                )
        if self.config.latent_dim % self.config.simnorm_dim != 0:
            raise ValueError(
                "Invalid TRM config for output SimNorm: "
                f"latent_dim={self.config.latent_dim} must be divisible by simnorm_dim={self.config.simnorm_dim}."
            )
        
        super().__init__()
        self.inner = TRMInner(self.config).to(torch.device('cuda'))
        self.use_act = self.config.halt_max_steps > 1
        self._halt_update_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = self._halt_update_eval
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.use_act:
            self._halt_update_fn = self._halt_update_train_act
        else:
            self._halt_update_fn = self._halt_update_eval
        return self

    def _halt_update_eval(self, new_steps: torch.Tensor, q_halt_logits: torch.Tensor) -> torch.Tensor:
        return new_steps >= self.config.halt_max_steps

    def _halt_update_train_act(self, new_steps: torch.Tensor, q_halt_logits: torch.Tensor) -> torch.Tensor:
        halted = (new_steps >= self.config.halt_max_steps) | (q_halt_logits > 0)
        random_steps = torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
        min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * random_steps
        return halted & (new_steps >= min_halt_steps)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device

        return TRMCarry(
            inner_carry=self.inner.empty_carry(batch_size, device=device),  # Empty is expected, it will be reset in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    @torch.compile(fullgraph=True, dynamic=True)
    def forward(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

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
            halted = self._halt_update_fn(new_steps, q_halt_logits)

        return TRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
