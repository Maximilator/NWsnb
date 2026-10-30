"""SDR-кодирование: LCA sparse coding (§II.1.1, Rozell 2008).

Плотный вектор → разреженный распределённый код (2048 нейронов, 2-5%
активных в каноне). LCA минимизирует E = ½‖x-Φa‖² + λΣ|a_i| через
локальную динамику конкурирующих нейронов:

  τ_du_i/dt = -u_i + b_i - Σ_{j≠i} G_ij a_j   (b=Φᵀx, G=ΦᵀΦ-I)
  a_i = T_λ(u_i) — мягкий порог

Обучение словаря Hebbian: ΔΦ = η_dict(x-Φa)aᵀ + нормировка колонок.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..config import EncoderCfg


class LCASparseEncoder:
    """Locally Competitive Algorithm — разреженный энкодер."""

    def __init__(self, input_dim: int, cfg: EncoderCfg, seed: int = 0):
        self.cfg = cfg
        self.n, self.d = cfg.n_sdr, input_dim
        g = torch.Generator().manual_seed(seed)
        self.phi = torch.randn(self.d, self.n, generator=g) / (self.d ** 0.5)
        self.u = torch.zeros(self.n)
        self.a = torch.zeros(self.n)

    def reset_state(self) -> None:
        self.u.zero_()
        self.a.zero_()

    def encode(self, x: torch.Tensor, steps: Optional[int] = None,
               learn: bool = True) -> torch.Tensor:
        """x [B,D] → SDR [B,N] (0/1)."""
        c = self.cfg
        n_steps = steps or c.lca_steps
        b = x @ self.phi                                  # [B,N]
        gram = self.phi.t() @ self.phi                    # [N,N]
        gram.fill_diagonal_(0.0)
        u = torch.zeros(x.shape[0], self.n, device=x.device)
        dt_tau = 1.0 / c.tau_lca
        for _ in range(n_steps):
            a = torch.relu(u - c.sparsity_lambda)         # soft threshold
            u = u + dt_tau * (-u + b - a @ gram)
        a = torch.relu(u - c.sparsity_lambda)
        # k-WTA до целевой доли — стабилизирует разреженность
        k = max(1, int(c.active_frac * self.n))
        thr = a.topk(k, dim=1).values[:, -1:]
        sdr = (a >= thr).float() * (a > 0).float()
        if learn:
            recon_err = x - a @ self.phi.t()
            dphi = recon_err.t() @ a / max(1, x.shape[0])
            self.phi += c.dict_lr * dphi
            self.phi /= self.phi.norm(dim=0, keepdim=True).clamp(min=1e-6)
        return sdr
