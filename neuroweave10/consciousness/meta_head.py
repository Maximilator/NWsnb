"""Мета-когнитивная голова HOT (§II.4.2).

Наблюдает GW и выдаёт:
  confidence     — «насколько система уверена в текущем решении»
  predicted_loss — L̂: предсказанная ошибка следующего шага
  surprise       — |фактический err - L̂| (мета-сюрприз)
  novelty        — расстояние GW-контента к эпизодической базе
  uncertainty    — декомпозиция epistemic (дисперсия по ансамблю
                   извлечений GW) / aleatoric (шум входа)

Обучение: L_meta = MSE + λ·Brier (калибровка уверенности).
M-ratio = meta-d'/d' — прокси метакогнитивной чувствительности.

Выход управляет поведением: confidence < θ_low гейтит действия,
увеличивает β_GW (крутизну ignition) и приоритет реплея.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..config import MetaHeadCfg


@dataclass
class MetaOut:
    confidence: float
    predicted_loss: float
    surprise: float
    novelty: float
    epistemic: float
    aleatoric: float
    m_ratio: float
    gates: Dict[str, float]  # → поведение/BG/TRN


class MetaHead:
    """MLP-голова над GW-содержимым + следы калибровки."""

    def __init__(self, cfg: MetaHeadCfg, d_gw: int, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.randn(cfg.d_hidden, d_gw, generator=g) * 0.1
        self.w2 = torch.randn(3, cfg.d_hidden, generator=g) * 0.1
        self.trace_gw = torch.zeros(d_gw)
        self.err_hist: List[float] = []
        self.conf_hist: List[float] = []

    # ------------------------------------------------------------------
    def _forward(self, gw: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(gw @ self.w1.t())
        return torch.sigmoid(h @ self.w2.t())   # [conf, L̂, nov]

    def step(self, gw_broadcast: torch.Tensor,
             gw_content: List[torch.Tensor],
             actual_error: float,
             reward: float = 0.0) -> MetaOut:
        c = self.cfg
        gw = gw_broadcast.detach().float()
        self.trace_gw = self.trace_gw * (1 - 1.0 / c.tau_trace) + gw
        out = self._forward(self.trace_gw)
        conf, l_hat, nov = float(out[0]), float(out[1]), float(out[2])
        # Эпистемическая неопределённость: дисперсия по ансамблю прогонов
        ens = [self._forward(gw + torch.randn_like(gw) * 0.05)
               for _ in range(c.n_ensemble)]
        epistemic = float(torch.stack(ens).var(0, unbiased=False).mean())
        # Алеаторная: шум входа — дисперсия по слотам
        if gw_content:
            aleatoric = float(torch.stack([x.float().flatten()
                                           for x in gw_content])
                              .var(0, unbiased=False).mean())
        else:
            aleatoric = 0.5
        surprise = abs(actual_error - l_hat)
        self.err_hist.append(actual_error)
        self.conf_hist.append(conf)
        # Обучение: MSE(L̂→err) + λ·Brier(conf→успех=1-err)
        target = torch.tensor([1.0 - min(1.0, actual_error),
                               min(1.0, actual_error), nov])
        loss = (out - target).pow(2).mean() + c.lambda_brier * \
            (out[0] - (1.0 - min(1.0, actual_error))).pow(2)
        # Ручной шаг SGD на последний слой (локальное правило)
        h = torch.tanh(self.trace_gw @ self.w1.t())
        grad_out = 2.0 * (out - target) * out * (1.0 - out)
        self.w2 -= c.lr * torch.outer(grad_out, h)
        self.w1 -= c.lr * torch.outer(self.w2.t() @ grad_out
                                      * (1 - h.pow(2)), self.trace_gw)
        # M-ratio: meta-d'/d' прокси — корр(conf, точность)
        if len(self.err_hist) > 8:
            import numpy as np
            c_arr = np.asarray(self.conf_hist[-64:])
            e_arr = 1.0 - np.asarray(self.err_hist[-64:])
            m = float(np.corrcoef(c_arr, e_arr)[0, 1]) \
                if c_arr.std() > 1e-6 and e_arr.std() > 1e-6 else 0.0
            m_ratio = m if np.isfinite(m) else 0.0
        else:
            m_ratio = 0.0
        gates = {
            "bg_threshold": 1.0 if conf < c.theta_low else 0.0,
            "beta_gw_boost": 0.3 if conf < c.theta_uncertain else 0.0,
            "replay_priority": epistemic,
            "trn_gain": 1.0 - conf * 0.5,
        }
        return MetaOut(confidence=conf, predicted_loss=l_hat,
                       surprise=surprise, novelty=nov,
                       epistemic=epistemic, aleatoric=aleatoric,
                       m_ratio=m_ratio, gates=gates)
