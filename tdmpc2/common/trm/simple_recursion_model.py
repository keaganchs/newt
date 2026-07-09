from typing import Dict, List, Tuple

from dataclasses import dataclass

from common.layers import mlp, SimNorm, NormedLinear, FiLM, latent_act
from common.trm.trm_layers import trunc_normal_init_, SwiGLU
from common.trm.dis_utils import advantage_margin, dis_beta, advantage_margin_curve_from_pending

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
        self.use_dis_loss = config.use_dis_loss
        self.dis_schedule = config.dis_schedule
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

        # Always create the scaffold (like SimpleTRM) so drain_grad_norms() is safe
        # even when logging is off -- tdmpc2.py drains "simple"/"srm" unconditionally.
        # forward() only touches it under `log_gradnorms and training`, so it stays
        # untouched (and empty) in the compiled, no-logging path below.
        self._pending_grad_norms: List[Dict[str, torch.Tensor]] = []
        if not self.log_gradnorms:
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

    def forward(self, carry: SRMCarry, z_star: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_star: stop-grad encoded next-state (the consistency-loss target), passed only
        during training. When None (inference, or all diagnostics/DIS off), no margin
        logging and no DIS loss is computed -- pure prediction. Returns (z, dis_loss),
        where dis_loss is a scalar (zero when DIS is off / z_star is None) carried back
        through the dataflow so it stays cudagraph-safe.
        """
        step_norms: Dict[str, torch.Tensor] = {}
        if self.log_gradnorms and self.training:
            self._pending_grad_norms.append(step_norms)

        # Diagnostics / DIS are train-time only and need z_star (never available at inference).
        log_margin = self.log_gradnorms and self.training and z_star is not None
        compute_dis = self.use_dis_loss and z_star is not None
        dis_loss = carry.z.new_zeros(())

        for h in range(self.H_cycles):
            # High recursion cycle: capture the carry before the context is refreshed so we
            # can log how far the high carry (z) moves and anchor the DIS intermediate target.
            z_before = carry.z
            carry.context = self._compute_context(carry.z)

            # First (L_cycles - T) iterations without gradients
            with torch.no_grad():
                for l in range(self.nograd_cycles):
                    z_inner_before = carry.z
                    carry = self._apply_fun(carry)
                    # Part 1: Advantage Margin at every inner iteration, including no-grad
                    # ones (advantage_margin detaches internally so this stays side-effect free).
                    if log_margin:
                        m_mean, m_frac = advantage_margin(z_inner_before, carry.z, z_star)
                        step_norms[f"h{h}_l{l}_advantage_margin_mean"] = m_mean
                        step_norms[f"h{h}_l{l}_advantage_margin_frac_nonpositive"] = m_frac

            # Last T iterations with gradients (truncated BPTT)
            for l in range(self.grad_cycles):
                z_inner_before = carry.z
                carry = self._apply_fun(carry)
                if log_margin:
                    gl = self.nograd_cycles + l
                    m_mean, m_frac = advantage_margin(z_inner_before, carry.z, z_star)
                    step_norms[f"h{h}_l{gl}_advantage_margin_mean"] = m_mean
                    step_norms[f"h{h}_l{gl}_advantage_margin_frac_nonpositive"] = m_frac

            # Part 2: DIS auxiliary loss on this high cycle's output, for s = h+1 in
            # 1..H_cycles-1. The final cycle (s = H_cycles, beta = 1.0) would target z_star
            # exactly, which the existing consistency loss already covers -> skipped here to
            # avoid double-counting. Gradient through carry.z respects the existing nograd/grad
            # cycle split; the only NEW detach is z_before (the moving-target anchor).
            if compute_dis and h < self.H_cycles - 1:
                s = h + 1
                beta = dis_beta(s, self.H_cycles, self.dis_schedule)
                z_dagger = (1.0 - beta) * z_before.detach() + beta * z_star
                dis_term = F.mse_loss(carry.z, z_dagger)
                dis_loss = dis_loss + dis_term
                if log_margin:
                    step_norms[f"y_{h}_dis_loss"] = dis_term.detach()

            # Log the high-carry delta/cossim once per high recursion cycle, using the
            # same metric names as SimpleTRM (y is its high carry; z is ours).
            if self.log_gradnorms and self.training:
                if h == self.H_cycles - 1 and carry.z.requires_grad:
                    carry.z.register_hook(lambda g: step_norms.__setitem__("y", g.detach().norm()))
                step_norms[f"y_{h}_delta"] = (carry.z.detach() - z_before.detach()).norm()
                step_norms[f"y_{h}_cossim"] = F.cosine_similarity(carry.z.detach(), z_before.detach(), dim=-1).mean()

        return carry.z, dis_loss

    def get_advantage_margin_curve(self) -> Dict[int, float]:
        """
        Mean Advantage Margin vs. global recursion-cycle index (h * L_cycles + l),
        aggregated from the pending gradnorm scaffold. Positive entries are cycles doing
        genuine refinement toward z*; entries near/below zero are dead compute. See
        Asadulaev et al. "Deep Improvement Supervision" for the underlying condition.
        """
        pending = getattr(self, "_pending_grad_norms", [])
        return advantage_margin_curve_from_pending(pending, self.config.L_cycles)

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
