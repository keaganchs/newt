from typing import List, Dict

from dataclasses import dataclass

from common.layers import mlp, SimNorm, NormedLinear, FiLM
from common.trm.trm_layers import trunc_normal_init_, SwiGLU

import torch
from torch import nn

from config import Config

@dataclass
class SRMCarry:
    x: torch.Tensor # [*batch, latent_dim + task_dim + action_dim]
    z: torch.Tensor # [*batch, latent_dim]
    context: torch.Tensor # [*batch, hidden_size]

    def detached(self) -> 'SRMCarry':
        return SRMCarry(self.x.detach(), self.z.detach(), self.context.detach())


class SRM(nn.Module):
    """
    Simple Recursive Model from https://www.researchsquare.com/article/rs-8492126/v1
    
    Similar to the TRM architecture but with a single carry state z and a context vector. This approach
    allows for reasoning at different timescales through skip connections, instead of relying on 
    hierarchical update cycles for separate carry states. 
    """
    def __init__(self, config: Config):
        super().__init__()
        # self.config = config
        # self.L_cycles = config.L_cycles
        self.H_cycles = config.H_cycles
        self.log_gradnorms = config.log_trm_gradnorms

        self.truncation_length = config.srm_truncation_length
        assert self.truncation_length <= config.L_cycles, f"Truncation length ({self.truncation_length}) cannot be greater than the total number of cycles ({config.L_cycles})."

        self.nograd_cycles = config.L_cycles - self.truncation_length
        self.grad_cycles = self.truncation_length






        self.net = mlp(config.latent_dim + config.task_dim + config.action_dim, [], config.latent_dim, act=SimNorm(config))
        
        
        # self.context_net = mlp(config.latent_dim, [], config.hidden_size, act=nn.Mish())
        self.context_net = nn.Sequential(
            SwiGLU(config.latent_dim, expansion=config.expansion),
            NormedLinear(config.latent_dim, config.hidden_size, act=nn.Mish())
        )


        if not self.log_gradnorms:
             self.forward = torch.compile(self.forward, fullgraph=True)


        pass

    def initial_carry(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        # z = trunc_normal_init_(torch.empty(*batch_shape, self.config.latent_dim, device=x.device, dtype=x.dtype), std=0.02)
        z = x[..., :self.config.latent_dim].detach().clone()  # warmup carry with the previous WM state
        
        context = self.context_net(z)
        # context = trunc_normal_init_(torch.empty(*batch_shape, self.config.hidden_size, device=x.device, dtype=x.dtype), std=0.02)
        
        return SRMCarry(x, z, context)


    def _apply_fun(self, carry: SRMCarry) -> SRMCarry:
        z = self.net(torch.cat([carry.z, carry.x + carry.context], dim=-1))
        return SRMCarry(carry.x, z, carry.context)

    def _compute_context(self, z: torch.Tensor) -> torch.Tensor:
        return self.context_net(z)


    def forward(self, carry: SRMCarry) -> torch.Tensor:
        # step_norms = {}

        # Reasoning steps
        for _ in range(self.H_cycles):
            carry.context = self._compute_context(carry.z)
        
            # First N-T cycles without gradients
            with torch.no_grad():
                for _ in range(self.nograd_cycles):
                    carry = self._apply_fun(carry)

            # Last T iterations with gradients
            for _ in range(self.grad_cycles):
                carry = self._apply_fun(carry)



            # Compute context from current carry state
            carry.context = self._compute_context(carry.z)
            

            for _ in range(self.L_cycles):
                carry = self._apply_fun(carry)








