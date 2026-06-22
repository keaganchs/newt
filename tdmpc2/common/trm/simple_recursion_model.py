from typing import Dict, List

from dataclasses import dataclass

from common.layers import mlp, SimNorm, NormedLinear, FiLM, latent_act
from common.trm.trm_layers import trunc_normal_init_, SwiGLU

import torch
from torch import nn
import torch.nn.functional as F

from config import Config


@dataclass
class SRMCarry:
    x: torch.Tensor        # [*batch, latent_dim + task_dim + action_dim]  (conditioning input, fixed for this step)
    z: torch.Tensor        # [*batch, latent_dim]                          (single recurrent state)
    context: torch.Tensor  # [*batch, hidden_size]                         (skip-connection signal, fnew(z0))

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
        self.config = config
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.log_gradnorms = config.log_trm_gradnorms
        self.use_film = config.use_film_dynamics

        self.truncation_length = config.srm_truncation_length
        assert self.truncation_length <= config.L_cycles, (
            f"Truncation length ({self.truncation_length}) cannot be greater than "
            f"the total number of cycles ({config.L_cycles})."
        )

        self.nograd_cycles = config.L_cycles - self.truncation_length
        self.grad_cycles = self.truncation_length

        x_dim = config.latent_dim + config.task_dim + config.action_dim

        # fnew: computes the skip-connection context from z0 at the start of each reasoning step
        self.context_net = nn.Sequential(
            SwiGLU(config.latent_dim, expansion=config.expansion),
            NormedLinear(config.latent_dim, config.hidden_size, act=nn.Mish()),
        )

        if self.use_film:
            # FiLM conditioning: context modulates x before it's combined with z
            # This keeps the skip connection "active" at every inner iteration
            self.film = FiLM(cond_dim=config.hidden_size, feature_dim=x_dim)
            net_input_dim = config.latent_dim + x_dim
        else:
            # Without FiLM: concatenate [z, x, context] directly.
            net_input_dim = config.latent_dim + x_dim + config.hidden_size

        # f: the shared recursive core.
        hidden_dims = [512, 512] if config.xl_dynamics_mlp else []
        self.net = mlp(
            net_input_dim,
            hidden_dims,
            config.latent_dim,
            act=latent_act(config),
        )

        self._init_weights()

        if self.log_gradnorms:
            self._pending_grad_norms: List[Dict[str, torch.Tensor]] = []
        else:
            self.forward = torch.compile(self.forward, fullgraph=True)

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_init_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initial_carry(self, x: torch.Tensor) -> SRMCarry:
        batch_shape = x.shape[:-1]
        z = x[..., :self.config.latent_dim].detach().clone()
        context = self.context_net(z)
        return SRMCarry(x, z, context)

    def _apply_fun(self, carry: SRMCarry) -> SRMCarry:
        if self.use_film:
            net_input = torch.cat([carry.z, self.film(carry.x, carry.context)], dim=-1)
        else:
            net_input = torch.cat([carry.z, carry.x, carry.context], dim=-1)
        z = self.net(net_input)
        return SRMCarry(carry.x, z, carry.context)

    def _compute_context(self, z: torch.Tensor) -> torch.Tensor:
        return self.context_net(z)

    def forward(self, carry: SRMCarry) -> torch.Tensor:
        step_norms: Dict[str, torch.Tensor] = {}
        if self.log_gradnorms and self.training:
            self._pending_grad_norms.append(step_norms)

        for h in range(self.H_cycles):
            carry.context = self._compute_context(carry.z)

            # First (L_cycles - T) iterations without gradients
            with torch.no_grad():
                for l in range(self.nograd_cycles):
                    z_before = carry.z
                    carry = self._apply_fun(carry)
                    if self.log_gradnorms and self.training:
                        name = f"y_h{h}_l{l}"
                        step_norms[f"{name}_delta"] = (carry.z - z_before).norm()
                        step_norms[f"{name}_cossim"] = F.cosine_similarity(carry.z, z_before, dim=-1).mean()

            # Last T iterations with gradients (truncated BPTT)
            for l in range(self.grad_cycles):
                z_before = carry.z
                carry = self._apply_fun(carry)

                if self.log_gradnorms and self.training:
                    name = f"y_h{h}_l{self.nograd_cycles + l}"
                    if carry.z.requires_grad:
                        carry.z.register_hook(
                            lambda g, name=name: step_norms.__setitem__(name, g.detach().norm())
                        )
                    step_norms[f"{name}_delta"] = (carry.z.detach() - z_before.detach()).norm()
                    step_norms[f"{name}_cossim"] = F.cosine_similarity(carry.z.detach(), z_before.detach(), dim=-1).mean()

        return carry.z

    def drain_grad_norms(self) -> List[Dict[str, torch.Tensor]]:
        result = self._pending_grad_norms
        self._pending_grad_norms = []
        return result

    def __repr__(self):
        lines = [f"SRM(H_cycles={self.H_cycles}, L_cycles={self.L_cycles}, truncation={self.truncation_length}, film={self.use_film})"]
        lines.append(f"  (net): {self.net}")
        if self.use_film:
            lines.append(f"  (film): {self.film}")
        lines.append(f"  (context_net): {self.context_net}")
        return "\n".join(lines)
