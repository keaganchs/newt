from typing import Tuple, List, Dict, Optional, Callable, Any
from dataclasses import dataclass

from common.layers import mlp, SimNorm 
from common.trm.trm_layers import trunc_normal_init_

import torch
from torch import nn

from config import Config


@dataclass
class SimpleTRMCarry:
    x: torch.Tensor # [*batch, latent_dim + task_dim + action_dim]
    y: torch.Tensor # [*batch, latent_dim]
    z: torch.Tensor # [*batch, latent_dim]


class SimpleTRM(nn.Module):
    """
    Simple TRM architecture for modeling dynamics in TD-MPC2.

    MLP layers have layer norm, with middle layers having a Mish activation and the final layer having a SimNorm activation.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        input_dim = self.config.task_dim + self.config.action_dim + 3 * self.config.latent_dim
        self.mlp = mlp(input_dim, [], self.config.latent_dim, act=SimNorm(self.config))

    def initial_carry(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        y = trunc_normal_init_(torch.empty(*batch_shape, self.config.latent_dim, device=x.device, dtype=x.dtype), std=0.02)
        z = trunc_normal_init_(torch.empty(*batch_shape, self.config.latent_dim, device=x.device, dtype=x.dtype), std=0.02)
        return SimpleTRMCarry(x, y, z)

    @torch.compile(fullgraph=True, dynamic=True)
    def apply_mlp(self, carry: SimpleTRMCarry) -> torch.Tensor:
        return self.mlp(torch.cat([carry.x, carry.y, carry.z], dim=-1))

    @torch.compile(fullgraph=True, dynamic=True)
    def forward(self, carry: SimpleTRMCarry):
        # Deep recursion; H_cycles-1 without grad
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                # Latent recursion
                for _ in range(self.config.L_cycles):
                    carry.z = self.apply_mlp(carry)
                carry.y = self.apply_mlp(carry)

        # Final pass with grad
        for _ in range(self.config.L_cycles):
            carry.z = self.apply_mlp(carry)
        carry.y = self.apply_mlp(carry)

        return carry.y
