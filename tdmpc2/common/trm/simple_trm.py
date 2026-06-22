from typing import List, Dict

from dataclasses import dataclass

from common.layers import mlp, SimNorm, NormedLinear, FiLM, latent_act
from common.trm.trm_layers import trunc_normal_init_, SwiGLU

import torch
from torch import nn
import torch.nn.functional as F

from config import Config


@dataclass
class SimpleTRMCarry:
    x: torch.Tensor # [*batch, latent_dim + task_dim + action_dim]
    y: torch.Tensor # [*batch, latent_dim]
    z: torch.Tensor # [*batch, latent_dim]

    def detached(self) -> 'SimpleTRMCarry':
        return SimpleTRMCarry(self.x.detach(), self.y.detach(), self.z.detach())


class SimpleTRM(nn.Module):
    """
    Simple TRM architecture for modeling dynamics in TD-MPC2.

    MLP layers have layer norm, with middle layers having a Mish activation and the final layer having a SimNorm activation.
    When use_film_dynamics=True, uses FiLM conditioning for the task embedding.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.use_film = config.use_film_dynamics
        self.use_skip = config.use_simple_trm_skip_connections
        self.skip_type = config.simple_trm_skip_type if self.use_skip else None
        self.mask_x_for_y = config.rrm_mask_x_for_y_update
        self.latent_dim = config.latent_dim

        if self.use_film:
            # Save values used for slicing the input in _apply_film
            self.task_dim = config.task_dim
            self.action_dim = config.action_dim

            # x = [z_initial | task_emb | action]; condition via FiLM on both task and action
            input_dim = config.latent_dim + 2 * config.latent_dim # wm dim + latent_dim for the y and z carries
            # hidden_mlp_dim = int(config.hidden_size * config.expansion)
            hidden_mlp_dim = self.latent_dim
            self.fc1 = NormedLinear(input_dim, hidden_mlp_dim)
            self.film = FiLM(config.task_dim + self.action_dim, hidden_mlp_dim)
            self.fc2 = NormedLinear(hidden_mlp_dim, config.latent_dim, act=latent_act(config))

            # self.fc1 = nn.Sequential(
            #     SwiGLU(input_dim, expansion=config.expansion),
            #     NormedLinear(input_dim, config.latent_dim, act=SimNorm(config))
            # )
            # self.film = FiLM(config.task_dim + self.action_dim, hidden_mlp_dim)
            # self.fc2 = nn.Sequential(
            #     SwiGLU(hidden_mlp_dim, expansion=config.expansion),
            #     NormedLinear(hidden_mlp_dim, config.latent_dim, act=SimNorm(config))
            # )


        else:
            input_dim = config.task_dim + config.action_dim + (3 * config.latent_dim) # wm dim + task + action in x, latent_dim for the y and z carries
            hidden_dims = [512, 512] if config.xl_dynamics_mlp else []
            self.mlp = mlp(input_dim, hidden_dims, config.latent_dim, act=latent_act(config))
            # self.mlp = nn.Sequential(
            #     SwiGLU(input_dim, expansion=config.expansion),
            #     NormedLinear(input_dim, config.latent_dim, act=SimNorm(config))
            # )

        if self.skip_type == "mlp":
            self.skip_mlp = mlp(config.latent_dim, [], config.latent_dim, act=nn.Mish())
        elif self.skip_type == "swiglu":
            self.skip_mlp = SwiGLU(hidden_size=config.latent_dim, expansion=self.config.expansion)
        self.log_trm_gradnorms = config.log_trm_gradnorms
        self._pending_grad_norms: List[Dict[str, torch.Tensor]] = []

        # Persistent bound-method reference so dynamo sees a stable callable identity
        # when tracing forward from an outer compiled context (e.g. loss_fn).
        self._apply_fn = self._apply_film if self.use_film else self._apply_mlp

    def initial_carry(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        # y = trunc_normal_init_(torch.empty(*batch_shape, self.config.hidden_size, device=x.device, dtype=x.dtype), std=0.02)
        y = x[..., :self.latent_dim].detach().clone()  # warmup carry on part of the input latent state (ablation: no random init, just use the input latent state directly)
        z = trunc_normal_init_(torch.empty(*batch_shape, self.config.latent_dim, device=x.device, dtype=x.dtype), std=0.02)
        return SimpleTRMCarry(x, y, z)

    def _apply_film(self, carry: SimpleTRMCarry, mask_x: bool = False) -> torch.Tensor:
        latent_dim = self.config.latent_dim
        task_dim = self.config.task_dim
        z_initial = carry.x[..., :latent_dim]
        task_emb = carry.x[..., latent_dim:latent_dim + task_dim]
        action = carry.x[..., latent_dim + task_dim:]
        cond = torch.cat([task_emb, action], dim=-1)
        if mask_x:
            z_initial = torch.zeros_like(z_initial)
        h = self.fc1(torch.cat([z_initial, carry.y, carry.z], dim=-1))
        return self.fc2(self.film(h, cond))

    def _apply_mlp(self, carry: SimpleTRMCarry, mask_x: bool = False) -> torch.Tensor:
        x = torch.zeros_like(carry.x) if mask_x else carry.x
        return self.mlp(torch.cat([x, carry.y, carry.z], dim=-1))

    def _skip(self, out: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
        if self.skip_type == "mlp" or self.skip_type == "swiglu":
            return out + self.skip_mlp(before)
        elif self.skip_type == "additive":
            return out + before
        else:
            raise ValueError(f"Unsupported skip type: {self.skip_type}")

    # Not compiled: forward is absorbed into the outer loss_fn compile (reduce-overhead).
    # Compiling _apply_* independently would create graph breaks in that outer trace.
    def forward(self, carry: SimpleTRMCarry):
        step_norms: Dict[str, torch.Tensor] = {}
        if self.log_trm_gradnorms and self.training:
            self._pending_grad_norms.append(step_norms)

        # H_cycles-1 warmup passes: detach outputs instead of using torch.no_grad(), so
        # grad_mode stays constant for _apply_fn across all call sites (no_grad would cause
        # dynamo to compile _apply_fn separately for each grad_mode state it encounters).
        # Passing z_before.detach() to _skip normalises before.requires_grad=False so that
        # guard never varies — the requires_grad_(True) workarounds are no longer needed.
        for h in range(self.config.H_cycles - 1):
            for _ in range(self.config.L_cycles):
                z_before = carry.z
                carry.z = self._apply_fn(carry).detach()
                if self.use_skip:
                    carry.z = self._skip(carry.z, z_before.detach())
            y_before = carry.y
            carry.y = self._apply_fn(carry, mask_x=self.mask_x_for_y).detach()
            if self.use_skip:
                carry.y = self._skip(carry.y, y_before.detach())
            if self.log_trm_gradnorms and self.training:
                step_norms[f"y_{h}_delta"] = (carry.y.detach() - y_before.detach()).norm()
                step_norms[f"y_{h}_cossim"] = F.cosine_similarity(carry.y.detach(), y_before.detach(), dim=-1).mean()

        # Final cycle with grad
        for i in range(self.config.L_cycles):
            z_before = carry.z
            carry.z = self._apply_fn(carry)
            if self.use_skip:
                carry.z = self._skip(carry.z, z_before)
            if self.log_trm_gradnorms and self.training:
                if carry.z.requires_grad:
                    carry.z.register_hook(lambda g, _i=i: step_norms.update({f"z_{_i}": g.detach().norm()}))
                step_norms[f"z_{i}_delta"] = (carry.z.detach() - z_before.detach()).norm()

        y_before = carry.y
        carry.y = self._apply_fn(carry, mask_x=self.mask_x_for_y)
        if self.use_skip:
            carry.y = self._skip(carry.y, y_before)
        if self.log_trm_gradnorms and self.training:
            h = self.config.H_cycles - 1
            if carry.y.requires_grad:
                carry.y.register_hook(lambda g: step_norms.update({"y": g.detach().norm()}))
            step_norms[f"y_{h}_delta"] = (carry.y.detach() - y_before.detach()).norm()
            step_norms[f"y_{h}_cossim"] = F.cosine_similarity(carry.y.detach(), y_before.detach(), dim=-1).mean()

        return carry.y

    def drain_grad_norms(self) -> List[Dict[str, torch.Tensor]]:
        result = self._pending_grad_norms
        self._pending_grad_norms = []
        return result

    def __repr__(self):
        lines = [f"SimpleTRM(H_cycles={self.config.H_cycles}, L_cycles={self.config.L_cycles}, skip={self.skip_type or 'none'})"]
        if self.use_film:
            lines.append(f"  (fc1): {self.fc1}")
            lines.append(f"  (film): {self.film}")
            lines.append(f"  (fc2): {self.fc2}")
        else:
            lines.append(f"  (mlp): {self.mlp}")
        if self.skip_type == "mlp" or self.skip_type == "swiglu":
            lines.append(f"  (skip_mlp): {self.skip_mlp}")
        return "\n".join(lines)
