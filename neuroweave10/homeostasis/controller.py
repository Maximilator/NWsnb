"""Гомеостатический контроллер (§1.5): σ≈1 во всех условиях.

Механизмы:
  MR-оценщик σ̂ → если σ>1: синаптическая депрессия (Levina),
                 ↓весов; σ<1: потенциация/нейрогенез;
  Intrinsic plasticity: ΔΘ = η_IP·(x - H_IP) — порог к целевой частоте;
  Vogels 2011 iSTDP: ингибиторные веса следят за пост-частотой ρ₀;
  SORN-нормировка: Σw_in ≈ const (синаптическое масштабирование).
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..config import HomeostasisCfg
from ..core.criticality import CriticalityMonitor


class HomeostaticController:
    """Регуляция критичности по всем зонам."""

    def __init__(self, cfg: HomeostasisCfg):
        self.cfg = cfg
        self.monitor = CriticalityMonitor(cfg.mr_kmax)
        self._prev_spikes: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def regulate(self, hierarchy, sigma: float,
                 dt_ms: float = 1.0) -> Dict[str, float]:
        """Применить регуляцию к весам/порогам всех колонок."""
        c = self.cfg
        report = {"sigma": sigma, "depression": 0.0, "ip": 0.0,
                  "istdp": 0.0}
        for col in hierarchy.cols:
            pool = col.pyr
            # 1) SOC-депрессия при σ>target+tol: сжать веса и ресурс
            if sigma > c.target_sigma + c.sigma_tol:
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    w.w *= 0.999
                    w.u_syn *= 0.999
                report["depression"] += 1.0
            elif sigma < c.target_sigma - c.sigma_tol:
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    w.w *= 1.001
            # 2) Intrinsic plasticity: Θ ← Θ + η_IP(x - H_IP)
            rate = pool.ca.mean().item() if hasattr(pool, "ca") else 0.0
            pool.theta += c.ip_rate * (rate - c.ip_target)
            report["ip"] += abs(c.ip_rate * (rate - c.ip_target))
            # 3) iSTDP Vogels: подтянуть ингибиторный баланс к ρ₀
            post = rate
            d_inh = c.inh_stdp_rate * (post - c.ip_target)
            col.w_pv_in += d_inh * 0.01
            col.w_pv_in.clamp_(0.0, 5.0)
            report["istdp"] += abs(d_inh)
            # 4) SORN-нормировка входа каждого нейрона (все три
            # проекции — иначе неконтролируемый канал растёт
            # экспоненциально под SOC-бустом).
            for w in (col.w_ff, col.w_fb, col.w_rec):
                nrm = w.w.norm(dim=1, keepdim=True).clamp(min=1e-6)
                w.w = w.w / nrm * 0.4
            # 5) Гомеостаз критичности → метаболический тормоз:
            # σ ниже цели — ослабить KATP/GIRK (меньше гиперполяризации,
            # сеть уходит из субкритичности); выше цели — усилить.
            if sigma < c.target_sigma - c.sigma_tol:
                pool.met_gain = max(0.1, getattr(pool, "met_gain", 1.0) * 0.995)
            elif sigma > c.target_sigma + c.sigma_tol:
                pool.met_gain = min(2.0, getattr(pool, "met_gain", 1.0) * 1.005)
            report["met_gain"] = float(getattr(pool, "met_gain", 1.0))
        return report
