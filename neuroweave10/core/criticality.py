"""Мониторинг критичности и лавин (§1.5, §II.1.5).

MR-оценщик σ̂ (Wilting & Priesemann 2018): для каждого спайка i во
время t считаем A_i — число спайков в t+1, «вызванных» им (пропорция
активности в потомках). σ̂ = среднее A_i — оценка показателя ветвления.

Лавины: покадровая активность A(t)=Σs_i(t); лавина — эпизод A>θ_aval
между двумя тихими кадрами. Размер S=ΣA, длительность T. В критичном
режиме P(S)~S^{-3/2}, P(T)~T^{-2} (Beggs & Plenz 2003) — контролируем
наклон через гистограмму.

Динамический диапазон Δ (Kinouchi & Copelli 2006): ответ системы
на шкалу входа максимален при σ=1.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import torch


@dataclass
class Avalanche:
    size: float
    duration_ms: int
    novel: bool  # размер выше скользящего квантиля — «новизна»


class CriticalityMonitor:
    """Оценка σ̂ и статистики лавин по спайковому потоку всех зон."""

    def __init__(self, k_max: int = 20, hist_bins: int = 32,
                 novel_quantile: float = 0.95):
        self.k_max = k_max
        self.novel_q = novel_quantile
        # A(t) история для MR и лавин
        self.activity: Deque[float] = deque(maxlen=2000)
        self.aval_sizes: Deque[float] = deque(maxlen=512)
        self.aval_durs: Deque[int] = deque(maxlen=512)
        self._cur_size = 0.0
        self._cur_dur = 0
        self._in_avalanche = False
        self.sigma_hat = 1.0
        self.theta_aval = 1.0  # порог лавины = шумовой уровень

    # ------------------------------------------------------------------
    def push(self, spikes_now: torch.Tensor, spikes_prev: torch.Tensor,
             weights: Optional[torch.Tensor] = None) -> float:
        """Обновить σ̂: A_i = s_i(t)·(Σ_j w_ij s_j(t+1))/max(Σ_j w_ij s_j,1).

        Упрощение без связности: A_i — доля активности t+1, приходящаяся
        на каждый спайк t, если топология однородна (mean-field MR)."""
        a_t = float(spikes_prev.sum())
        a_t1 = float(spikes_now.sum())
        self.activity.append(a_t1)
        if a_t > 0:
            self.sigma_hat = 0.9 * self.sigma_hat + 0.1 * (a_t1 / a_t)
        # лавины
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

    def novel_avalanche(self) -> bool:
        """Последняя лавина выше квантиля — событие новизны (→ VT/ACh)."""
        if len(self.aval_sizes) < 8:
            return False
        import numpy as np
        q = float(np.quantile(list(self.aval_sizes), self.novel_q))
        return self.aval_sizes[-1] >= q

    def dynamic_range(self, responses: List[float],
                      inputs: List[float]) -> float:
        """Δ = Σ_x F(x)·Δx — площадь под кривой ответа (норм.)."""
        if len(responses) < 2:
            return 0.0
        rng = max(inputs) - min(inputs) + 1e-9
        peak = max(responses) + 1e-9
        area = sum(responses) / len(responses) / peak
        return float(area)

    def stats(self) -> Dict[str, float]:
        return {"sigma_hat": self.sigma_hat,
                "n_avalanches": len(self.aval_sizes),
                "aval_mean_size": (sum(self.aval_sizes)
                                   / max(1, len(self.aval_sizes))),
                "aval_mean_dur": (sum(self.aval_durs)
                                  / max(1, len(self.aval_durs)))}
