"""Синапс трёхфакторного обучения (§1.3).

Правило Urbanczik-Senn + модуляторный фактор:

  dw_ij/dt = η · f_mod(M) · [S_i - φ(V*_D,i)] · x_j
  θ_m(t)   = θ_0 + γ_GC[GC] + γ_E2[E2] + γ_H2S[H2S] + γ_Zn[Zn2+] + γ_Mg[Mg2+]

  f_mod = (α_DA[DA] + β_NE[NE])·(1 - κ_eCB[2-AG])·Φ_CORT
          ·H(E-θ_E)·H(θ_S - S)

θ_m — метапластический порог BCM с химическими членами:
глюкокортикоиды закрывают окно пластичности, эстрадиол/H2S/Zn/Mg
модулируют его. Хранится как O(1) буфер на синапс — третий фактор
единообразен, не N×M.

Eligibility-экспонента: z_pre/z_post — O(N) следы (Izhikevich 2007);
произведение x_j·err_i вычисляется внешним произведением только по
активным маскам — разреженный путь.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import USSynapseCfg


@dataclass
class SynOut:
    w_mean: float
    dw_mean: float
    ltp_mask: torch.Tensor   # гетеросинаптическая LTP (NO-окрестность)


class GenesisSynapse:
    """Двунаправленный синапс: feedforward-вес + следы + метапластичность."""

    def __init__(self, n_pre: int, n_post: int, cfg: USSynapseCfg,
                 name: str = "syn", seed: int = 0, init_scale: float = 0.4):
        self.cfg, self.name = cfg, name
        g = torch.Generator().manual_seed(seed)
        self.w = (torch.rand(n_post, n_pre, generator=g) * init_scale
                  - init_scale / 2.0) \
            * (1.0 if cfg.sign == "exc" else -1.0)
        self.mask = (torch.rand(n_post, n_pre, generator=g) < cfg.density)
        self.w = self.w * self.mask
        self.z_pre = torch.zeros(n_pre)     # x_j — пре-след
        self.z_post = torch.zeros(n_post)   # пост-след (для iSTDP)
        self.theta_m = cfg.theta_m0         # метапластический порог BCM
        self.u_syn = torch.ones(n_post, n_pre)  # ресурс депрессии (SOC)

    # ------------------------------------------------------------------
    def set_metaplastic(self, gc: float, e2: float, h2s: float,
                        zn: float, mg: float) -> None:
        """θ_m = θ_0 + γ_GC[GC] + γ_E2[E2] + γ_H2S[H2S] + γ_Zn[Zn] + γ_Mg[Mg]."""
        self.theta_m = self.cfg.theta_m0 + 0.5 * gc - 0.3 * e2 \
            - 0.2 * h2s + 0.2 * zn - 0.1 * mg

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """Проводимость: g = (x @ W_eff), W_eff = w·mask·u_syn."""
        return x @ (self.w * self.mask * self.u_syn).t()

    def update_traces(self, s_pre: torch.Tensor, s_post: torch.Tensor,
                      dt: float = 1.0) -> None:
        c = self.cfg
        self.z_pre = self.z_pre * (1.0 - dt / c.tau_x) \
            + s_pre.mean(0)  # среднее по батчу — общий след
        self.z_post = self.z_post * (1.0 - dt / c.tau_x) \
            + s_post.mean(0)

    def learn(self,
              error: torch.Tensor,      # [B,N_post] S_i - φ(V*_D)
              s_pre: torch.Tensor,      # [B,N_pre]
              s_post: torch.Tensor,     # [B,N_post]
              f_mod: float,             # химический гейт (скаляр)
              ltp_field: Optional[torch.Tensor] = None,  # NO-окрестность [N_pre]
              dt: float = 1.0) -> SynOut:
        """Трёхфакторное обновление. Всё без градиентов."""
        c = self.cfg
        err = error.mean(0)                     # [N_post]
        x = s_pre.mean(0)                       # [N_pre]
        if f_mod <= 0.0:
            return SynOut(float(self.w.abs().mean()), 0.0,
                          torch.zeros_like(x))
        # Δw = η·f_mod·err_i·x_j — по маске, с метапластическим гейтом θ_m.
        # BCM: пост-активность выше θ_m → LTP; ниже → ослабленная LTD-половина.
        post_rate = s_post.mean(0)
        bcm_sign = torch.where(post_rate > self.theta_m, 1.0, -0.5)
        dw = c.eta * f_mod * torch.outer(err * bcm_sign, x) * self.mask
        # Гетеросинаптическая LTP: NO-поле помечает соседние терминали
        if ltp_field is not None:
            dw = dw + c.eta * f_mod * 0.1 * torch.outer(
                err.abs(), ltp_field) * self.mask
        self.w = (self.w + dw).clamp(c.w_min, c.w_max)
        # Медленная утечка к нулю (забывание)
        self.w = self.w * c.decay
        # SOC-депрессия: спайк тратит ресурс u_syn
        spent = torch.outer(s_post.mean(0), x).clamp(0.0, 1.0) * self.mask
        self.u_syn = (self.u_syn - spent * 0.15).clamp(min=0.05)
        self.u_syn += (1.0 - self.u_syn) * (dt / 500.0)  # восстановление
        self.update_traces(s_pre, s_post, dt)
        return SynOut(float(self.w.abs().mean()), float(dw.abs().mean()),
                      x)
