"""Базальные ганглии (§II.3.3): DDM + Go/NoGo/STN + TD-RPE.

DDM-выбор:
  dx_i/dt = (w_Go,i - w_NoGo,i)·I_ctx - leak·x_i + ξ;  x_i(0)=0
  a(t)    = a_0 + k_STN·H(softmax(v))     — конфликт ↑ порог
            + α_OX·(-[OX]) + α_ADO·[ADO] + 5-HT·θ_HT
  Первый x_i > a(t) → действие; STN приходит за 5-10 мс до решения
  (гиперпрямой путь → пауза при конфликте).

Обучение (Frank 2005):
  δ = r + γ·V(s') - V(s)                    — TD-ошибка → фазический DA
  Δw_Go   = α·δ·e_pre·[D1]    (δ>0 → LTP прямого пути)
  Δw_NoGo = α·(-δ)·e_pre·[D2] (δ<0 → LTP непрямого пути)
  A2A-D2 гетеромер: ADO ослабляет D2 → сдвиг баланса к Go.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import BasalGangliaCfg


@dataclass
class BGOut:
    action: torch.Tensor      # [B] выбранный индекс или -1
    decided: bool
    decision_ms: float
    conflict: float
    threshold: float
    value: torch.Tensor       # V(s) критика


class BasalGanglia:
    """Go/NoGo DDM + STN + TD-критик."""

    def __init__(self, cfg: BasalGangliaCfg, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.w_go = torch.randn(cfg.n_actions, cfg.n_actions, generator=g) * 0.1
        self.w_nogo = torch.randn(cfg.n_actions, cfg.n_actions,
                                  generator=g) * 0.1
        self.w_v = torch.zeros(cfg.n_actions)   # критик V(s)
        self.acc = torch.zeros(cfg.n_actions)
        self.e_pre = torch.zeros(cfg.n_actions) # eligibility след
        self.v_prev = 0.0
        self._pending_ms = -1.0

    # ------------------------------------------------------------------
    def threshold(self, v: torch.Tensor, ox: float, ado: float,
                  ht: float) -> float:
        """a(t) = a_0 + k_STN·conflict - α_OX·OX + α_ADO·ADO + θ_5HT."""
        c = self.cfg
        p = torch.softmax(v * 4.0, dim=-1)
        conflict = float(-(p * p.clamp(min=1e-8).log()).sum() /
                         torch.log(torch.tensor(float(len(p)))))
        return c.a0 + c.k_stn * conflict - c.alpha_ox * ox \
            + c.alpha_ado * ado + c.ht_patience * ht

    def step(self, ctx: torch.Tensor, ox: float, ado: float,
             ht: float, da_d2_shift: float = 0.0) -> BGOut:
        """ctx [B,N] — кортикальный контекст (PFC)."""
        c = self.cfg
        # Кортикостриатарная проекция: ширина весов = ширине PFC-входа
        # (n_actions — число аккумуляторов решений, не обязан
        # совпадать с числом нейронов PFC).
        if self.w_go.shape[1] != ctx.shape[-1]:
            g = torch.Generator().manual_seed(11)
            self.w_go = torch.randn(c.n_actions, ctx.shape[-1],
                                    generator=g) * 0.1
            self.w_nogo = torch.randn(c.n_actions, ctx.shape[-1],
                                      generator=g) * 0.1
        v = ctx @ (self.w_go - self.w_nogo).t()          # [B,N] дрейф
        v = v.mean(0)
        # А2А-D2: аденозин ослабляет D2 → NoGo-путь слабее
        v = v + c.k_het * ado * (ctx @ self.w_nogo.t()).mean(0) * -0.5
        noise = torch.randn(c.n_actions) * c.sigma_ddm
        self.acc = self.acc + (v - c.leak * self.acc) + noise
        a_t = self.threshold(v, ox, ado, ht)
        win = self.acc.argmax()
        decided = bool(self.acc[win] > a_t)
        p = torch.softmax(v * 4.0, dim=-1)
        conflict = float(-(p * p.clamp(min=1e-8).log()).sum()
                         / torch.log(torch.tensor(float(len(p)))))
        action = torch.tensor([win if decided else -1])
        return BGOut(action=action, decided=decided,
                     decision_ms=0.0, conflict=conflict,
                     threshold=a_t, value=self.w_v.clone())

    def reward(self, r: float, ctx_mean: torch.Tensor, da: float,
               ado: float) -> float:
        """TD-RPE: δ = r + γV' - V → обновляет w_go/w_nogo/w_v.

        Возвращает δ (фазический DA импульс в химию)."""
        c = self.cfg
        v_now = float(ctx_mean @ self.w_v) if ctx_mean.numel() else 0.0
        delta = r + c.gamma_td * v_now - self.v_prev
        self.v_prev = v_now
        self.e_pre = self.e_pre * 0.95 + ctx_mean.detach()[:c.n_actions] \
            if ctx_mean.numel() >= c.n_actions else self.e_pre
        d2 = (1.0 - c.k_het * ado)          # A2A-D2 ослабление
        self.w_go += c.alpha_rl * delta * torch.outer(
            self.e_pre, self.e_pre) * da
        self.w_nogo += c.alpha_rl * (-delta) * torch.outer(
            self.e_pre, self.e_pre) * d2
        if ctx_mean.numel() >= c.n_actions:
            self.w_v += c.alpha_rl * delta * ctx_mean[:c.n_actions]
        return float(delta)
