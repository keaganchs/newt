"""
Diagnostics and auxiliary losses for the recursive dynamics models, ported to the
continuous-latent setting from Asadulaev et al. "Deep Improvement Supervision" (DIS).

Two ideas live here:

  * Advantage Margin (diagnostic only). In the discrete-token version each recursion
    step produces pre-/post-update logits and the margin measures whether the step
    moved probability mass toward the correct answer more than toward the average
    alternative. With no vocabulary we use the existing stop-grad consistency target
    z* and measure how much closer the carry got to it across a cycle:
        A_s = ||z^(s-1) - z*||^2 - ||z^(s) - z*||^2
    A_s > 0 ==> genuine refinement; A_s <= 0 ==> dead compute for that cycle.

  * The interpolation schedule (beta_s) used by the DIS auxiliary loss to build the
    moving intermediate targets z_dagger_s = (1 - beta_s) * z_prev + beta_s * z*.

Everything here is train-time only: it needs z* (the consistency-loss target), which
is never available at inference, so callers gate it on z_star being passed.
"""
from typing import Dict, List, Tuple

import torch


def advantage_margin(
    z_prev: torch.Tensor,
    z_curr: torch.Tensor,
    z_star: torch.Tensor,
    eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-cycle Advantage Margin A_s = ||z_prev - z*||^2 - ||z_curr - z*||^2.

    All inputs are detached internally so this can never perturb the gradient graph
    (the diagnostic must be free of side effects on training). Returns a pair of
    scalar tensors: (mean margin over the batch, fraction of the batch with
    A_s <= eps, i.e. the share of "dead"/non-improving examples for this cycle).
    See Asadulaev et al. "Deep Improvement Supervision" for the Advantage Margin
    condition this approximates.
    """
    with torch.no_grad():
        z_prev = z_prev.detach()
        z_curr = z_curr.detach()
        z_star = z_star.detach()
        dist_prev = (z_prev - z_star).pow(2).sum(-1)
        dist_curr = (z_curr - z_star).pow(2).sum(-1)
        margin = dist_prev - dist_curr  # shape: batch (positive == improvement)
        mean = margin.mean()
        frac_nonpositive = (margin <= eps).float().mean()
    return mean, frac_nonpositive


def dis_beta(s: int, num_cycles: int, schedule: str = "linear") -> float:
    """
    Interpolation coefficient beta_s for high-cycle index s in 1..num_cycles.

    Monotonically increasing from ~0 toward 1 with beta_{num_cycles} == 1.0 exactly,
    so the final cycle's intermediate target equals z* (matching the existing
    consistency loss). `schedule` is one of "linear" (default) or "cosine".
    """
    assert 1 <= s <= num_cycles, f"cycle index {s} out of range 1..{num_cycles}"
    if num_cycles <= 1:
        return 1.0
    frac = s / num_cycles  # frac == 1.0 exactly at s == num_cycles
    if schedule == "linear":
        return frac
    elif schedule == "cosine":
        # Ease-in from 0 to 1; cos(pi/2 * (1 - frac)) hits 1.0 exactly at frac == 1.
        import math
        return math.cos((math.pi / 2.0) * (1.0 - frac))
    else:
        raise ValueError(f"Unknown dis_schedule: {schedule}")


def advantage_margin_curve_from_pending(
    pending: List[Dict[str, torch.Tensor]],
    l_cycles: int,
) -> Dict[int, float]:
    """
    Aggregate the per-cycle advantage-margin entries stashed in the gradnorm
    `step_norms` scaffold into a single mean-margin-vs-cycle curve.

    Scans every pending step_norms dict (one per dynamics call in the rollout) for
    keys of the form ``h{H}_l{L}_advantage_margin_mean`` and averages them, keyed by
    the flattened global cycle index ``H * l_cycles + L``. Intended for plotting where
    in the recursion the margin collapses to ~0 after a run; see Asadulaev et al.
    "Deep Improvement Supervision".
    """
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    suffix = "_advantage_margin_mean"
    for step_norms in pending:
        for key, val in step_norms.items():
            if not key.endswith(suffix):
                continue
            core = key[: -len(suffix)]  # e.g. "h2_l1"
            try:
                h_part, l_part = core.split("_")
                h = int(h_part[1:])
                l = int(l_part[1:])
            except (ValueError, IndexError):
                continue
            global_idx = h * l_cycles + l
            v = val.item() if isinstance(val, torch.Tensor) else float(val)
            sums[global_idx] = sums.get(global_idx, 0.0) + v
            counts[global_idx] = counts.get(global_idx, 0) + 1
    return {idx: sums[idx] / counts[idx] for idx in sorted(sums)}
