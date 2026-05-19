from typing import Tuple, List, Dict, Optional, Callable, Any
from dataclasses import dataclass

from common.layers import mlp, SimNorm, NormedLinear, FiLM
from common.trm.trm_layers import trunc_normal_init_

import torch
from torch import nn

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
    When use_film_dynamics=True, uses FiLM conditioning for the task embedding
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.use_film = config.use_film_dynamics

        if self.use_film:
            # x = [z_initial | task_emb | action]; condition via FiLM, not concatenation
            input_no_task_dim = config.action_dim + 3 * config.latent_dim
            self.fc1 = NormedLinear(input_no_task_dim, config.mlp_dim)
            self.film = FiLM(config.task_dim, config.mlp_dim)
            self.fc2 = NormedLinear(config.mlp_dim, config.latent_dim, act=SimNorm(config))
        else:
            input_dim = config.task_dim + config.action_dim + 3 * config.latent_dim
            self.mlp = mlp(input_dim, [], config.latent_dim, act=SimNorm(config))

        # For logging grad norms through each backward pass/update call through the recursion steps
        self._pending_grad_norms: List[Dict[str, torch.Tensor]] = []

    def initial_carry(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        y = trunc_normal_init_(torch.empty(*batch_shape, self.config.latent_dim, device=x.device, dtype=x.dtype), std=0.02)
        z = trunc_normal_init_(torch.empty(*batch_shape, self.config.latent_dim, device=x.device, dtype=x.dtype), std=0.02)
        return SimpleTRMCarry(x, y, z)

    # Apply FiLM layers
    @torch.no_grad()
    @torch.compile(fullgraph=True, dynamic=True)
    def _apply_film_nograd(self, carry: SimpleTRMCarry) -> torch.Tensor:
        latent_dim = self.config.latent_dim
        task_dim = self.config.task_dim
        task_emb = carry.x[..., latent_dim:latent_dim + task_dim]
        x_no_task = torch.cat([carry.x[..., :latent_dim], carry.x[..., latent_dim + task_dim:]], dim=-1)
        h = self.fc1(torch.cat([x_no_task, carry.y, carry.z], dim=-1))
        return self.fc2(self.film(h, task_emb))

    @torch.compile(dynamic=True)
    def _apply_film_grad(self, carry: SimpleTRMCarry) -> torch.Tensor:
        latent_dim = self.config.latent_dim
        task_dim = self.config.task_dim
        task_emb = carry.x[..., latent_dim:latent_dim + task_dim]
        x_no_task = torch.cat([carry.x[..., :latent_dim], carry.x[..., latent_dim + task_dim:]], dim=-1)
        h = self.fc1(torch.cat([x_no_task, carry.y, carry.z], dim=-1))
        return self.fc2(self.film(h, task_emb))

    # Apply MLP layers
    @torch.no_grad()
    @torch.compile(fullgraph=True, dynamic=True)
    def _apply_mlp_nograd(self, carry: SimpleTRMCarry) -> torch.Tensor:
        return self.mlp(torch.cat([carry.x, carry.y, carry.z], dim=-1))

    @torch.compile(dynamic=True)
    def _apply_mlp_grad(self, carry: SimpleTRMCarry) -> torch.Tensor:
        return self.mlp(torch.cat([carry.x, carry.y, carry.z], dim=-1))

    # Not compiled: register_hook on intermediate tensors requires eager execution.
    # _apply_*_nograd / _apply_*_grad are still compiled.
    def forward(self, carry: SimpleTRMCarry):
        if self.use_film:
            apply_nograd = self._apply_film_nograd
            apply_grad = self._apply_film_grad
        else:
            apply_nograd = self._apply_mlp_nograd
            apply_grad = self._apply_mlp_grad

        # H_cycles-1 warm-up passes without grad. The @torch.no_grad() decorator on apply_nograd
        # handles the no-grad context; carry.detached() ensures requires_grad=False on all inputs.
        for _ in range(self.config.H_cycles - 1):
            for _ in range(self.config.L_cycles):
                carry.z = apply_nograd(carry.detached())
            carry.y = apply_nograd(carry.detached())

        # During inference/planning (no grad context), use nograd variant for the final pass too,
        # so apply_grad is never called with grad_mode=False (which would trigger recompilation).
        if not torch.is_grad_enabled():
            for _ in range(self.config.L_cycles):
                carry.z = apply_nograd(carry.detached())
            carry.y = apply_nograd(carry.detached())
            return carry.y

        # Ensure carry.z has requires_grad=True before the first call so apply_grad always sees
        # the same guard state (and doesn't trigger recompilation).
        carry.z = carry.z.detach().requires_grad_(True)
        step_norms: Dict[str, torch.Tensor] = {}
        if self.training:
            self._pending_grad_norms.append(step_norms)

        # Final pass with grad; register hooks to track per-step gradient norms.
        for i in range(self.config.L_cycles):
            carry.z = apply_grad(carry)
            if self.training and carry.z.requires_grad:
                carry.z.register_hook(
                    lambda g, _i=i: step_norms.update({f"z_{_i}": g.detach().norm()})
                )
        carry.y = apply_grad(carry)
        if self.training and carry.y.requires_grad:
            carry.y.register_hook(lambda g: step_norms.update({"y": g.detach().norm()}))

        return carry.y
