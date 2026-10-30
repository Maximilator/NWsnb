"""Мониторинг критичности и лавин (§1.5, §II.1.5; math.md §5).

MR-оценщик σ̂ (Wilting & Priesemann 2018): автокорреляции
r_k = m^k → фит log r_k = k·log m̂ по истории активности
(устойчив к субсэмплингу). Целевой режим по in-vivo данным:
σ̂ ≈ 0.95-0.99 («ревербирующий», слегка субкритичный), τ_int=Δt/(1−m̂)
конечен и перестраиваем — не σ=1.

Лавины: эпизод A(t)>θ_aval между тихими кадрами; P(S)~S^{-3/2},
P(T)~T^{-2}; новизна — лавина выше скользящего квантиля.
Динамический диапазон: Δ = 10·log10(F_0.9/F_0.1) (Kinouchi-Copelli).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

import torch


@dataclass
class Avalanche:
    size: float
    duration_ms: int
    novel: bool


class CriticalityMonitor:
    """σ̂ по автокорреляционной регрессии + статистика лавин."""

    def __init__(self, k_max: int = 20, novel_quantile: float = 0.95,
                 target_sigma: float = 0.97):
        self.k_max = k_max
        self.novel_q = novel_quantile
        self.target = target_sigma
        self.activity: Deque[float] = deque(maxlen=2000)
        self.aval_sizes: Deque[float] = deque(maxlen=512)
        self.aval_durs: Deque[int] = deque(maxlen=512)
        self._cur_size, self._cur_dur = 0.0, 0
        self._in_avalanche = False
        self.sigma_hat = target_sigma
        self.theta_aval = 1.0

    def push(self, spikes_now: torch.Tensor,
             spikes_prev: torch.Tensor) -> float:
        a_t1 = float(spikes_now.sum())
        self.activity.append(a_t1)
        if len(self.activity) > self.k_max + 2:
            self.sigma_hat = self._mr_estimate()
        if a_t1 > self.theta_aval:
            if not self._in_avalanche:
                self._in_avalanche, self._cur_size, self._cur_dur = True, 0.0, 0
            self._cur_size += a_t1
            self._cur_dur += 1
        elif self._in_avalanche:
            self._in_avalanche = False
            self.aval_sizes.append(self._cur_size)
            self.aval_durs.append(self._cur_dur)
        return self.sigma_hat

    def _mr_estimate(self) -> float:
        """r_k = автокорреляция на лаге k; fit log r_k = k log m."""
        import numpy as np
        a = np.asarray(list(self.activity), dtype=float)
        a = a - a.mean()
        v = (a * a).mean() + 1e-12
        rs, ks = [], []
        for k in range(1, min(self.k_max, len(a) - 1) + 1):
            r_k = float((a[:-k] * a[k:]).mean() / v)
            if r_k > 1e-6:
                rs.append(np.log(r_k)); ks.append(k)
        if len(ks) < 2:
            return self.sigma_hat
        slope = float(np.polyfit(ks, rs, 1)[0])
        return float(np.clip(np.exp(slope), 0.0, 3.0))

    def novel_avalanche(self) -> bool:
        if len(self.aval_sizes) < 8:
            return False
        import numpy as np
        return float(self.aval_sizes[-1]) >= float(
            np.quantile(list(self.aval_sizes), self.novel_q))

    def dynamic_range(self, responses: List[float],
                      inputs: List[float]) -> float:
        """Δ = 10·log10(F(0.9)/F(0.1)) (Kinouchi & Copelli 2006)."""
        if len(responses) < 4:
            return 0.0
        import numpy as np
        r = np.sort(np.asarray(responses, dtype=float))
        f90, f10 = r[int(0.9 * (len(r) - 1))], r[int(0.1 * (len(r) - 1))]
        return float(10.0 * np.log10((f90 + 1e-9) / (f10 + 1e-9)))

    def stats(self) -> Dict[str, float]:
        return {"sigma_hat": self.sigma_hat,
                "n_avalanches": len(self.aval_sizes),
                "aval_mean_size": (sum(self.aval_sizes)
                                   / max(1, len(self.aval_sizes))),
                "aval_mean_dur": (sum(self.aval_durs)
                                  / max(1, len(self.aval_durs)))}
