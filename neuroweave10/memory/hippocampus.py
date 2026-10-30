"""Гиппокампальный комплекс (§II.3.1): EC → DG → CA3 → CA1 → VTA/LC.

DG — паттерн-сепарация: экспансия + разрежение (~1-2% активных),
взрослый нейрогенез (2-3% нейронов/месяц → реинтеграция весов).
CA3 — автоассоциативная сеть Хопфилда, one-shot обучение
(η_CA3 = 10·η_cortex, ёмкость ~0.14·N).
CA1 — компаратор: mismatch ||CA3-EC_III||² → фазический DA (VTA)
и NE (LC); novelty-сигнал.
SWR-реплей: [ACh]<θ & [Adenosine]>θ → prioritized experience replay,
P(replay) ∝ |ошибка| + β·uncertainty (HOT), прокси-градиент
на кортикальные веса.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ..config import HippocampusCfg


@dataclass
class HippoOut:
    ec: torch.Tensor
    dg: torch.Tensor
    ca3: torch.Tensor
    ca1_out: torch.Tensor
    mismatch: float
    vta_da_burst: float      # фазический DA от новизны
    lc_ne_burst: float       # фазический NE от новизны
    replayed: List[int]      # индексы реплеированных эпизодов


class Hippocampus:
    """Трисинаптический контур с нейрогенезом и SWR-реплеем."""

    def __init__(self, cfg: HippocampusCfg, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.w_ec_dg = torch.randn(cfg.n_dg, cfg.n_ec, generator=g) * 0.3
        self.w_dg_ca3 = torch.randn(cfg.n_ca3, cfg.n_dg, generator=g) * 0.1
        self.w_ca3 = torch.zeros(cfg.n_ca3, cfg.n_ca3)   # Хопфилд
        self.w_ca3_ca1 = torch.randn(cfg.n_ec, cfg.n_ca3, generator=g) * 0.1
        self.dg_age = torch.ones(cfg.n_dg)               # возраст гранул
        self.tick_n = 0
        # Эпизодический буфер: (ec, dg, ca3, ошибка, uncertainty)
        self.episodes: List[Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, float, float]] = []
        self.max_episodes = 512

    # ------------------------------------------------------------------
    def dg_code(self, ec: torch.Tensor) -> torch.Tensor:
        """Паттерн-сепарация: разрежение до dg_sparsity (k-WTA)."""
        drive = ec @ self.w_ec_dg.t()
        k = max(1, int(self.cfg.dg_sparsity * self.cfg.n_dg))
        thr = drive.topk(k, dim=1).values[:, -1:]
        return (drive >= thr).float()

    def neurogenesis(self) -> None:
        """Взрослый нейрогенез DG: замена ~2% старейших гранул.

        Новые нейроны гиперэксайтабельны (молодые гранулы легче в DG)
        — сброс входных весов у старейших, «возраст» → 0."""
        c = self.cfg
        n_new = max(1, int(0.02 * c.n_dg))
        oldest = self.dg_age.argsort(descending=True)[:n_new]
        self.w_ec_dg[oldest] = torch.randn(n_new, c.n_ec) * 0.5
        self.dg_age[oldest] = 0.0
        self.dg_age += 1.0

    def ca3_store(self, ca3: torch.Tensor) -> None:
        """One-shot Хопфилд: w += η_CA3·(ca3⊗ca3 - I), ёмкость 0.14N."""
        c = self.cfg
        p = ca3.mean(0)
        dw = torch.outer(p, p) - torch.diag(p.pow(2))
        self.w_ca3 += c.eta_ca3 * 10.0 * dw
        self.w_ca3.clamp_(-1.0, 1.0)

    def ca3_retrieve(self, cue: torch.Tensor) -> torch.Tensor:
        """Pattern completion: несколько итераций рекуррентной динамики."""
        a = cue.clone()
        for _ in range(self.cfg.ca3_steps):
            a = torch.tanh(a @ self.w_ca3.t() * 2.0)
        return (a > 0).float()

    def step(self, ec_in: torch.Tensor, ach: float, adenosine: float,
             learn: bool = True) -> HippoOut:
        """Бодрствование: кодирование; сон (ACh↓,Ado↑): SWR-реплей."""
        self.tick_n += 1
        if self.tick_n % self.cfg.neurogenesis_period == 0:
            self.neurogenesis()
        ec = ec_in
        dg = self.dg_code(ec)
        ca3_raw = torch.relu(dg @ self.w_dg_ca3.t())
        ca3 = ca3_raw
        if learn:
            self.ca3_store(ca3)
        ca3_full = self.ca3_retrieve(ca3)
        # CA1-компаратор: реконструкция vs энторинальный вход
        recon = ca3_full @ self.w_ca3_ca1.t()
        mismatch = float((recon - ec).pow(2).mean())
        burst_da = float(mismatch > self.cfg.ca1_threshold) * min(
            1.0, mismatch / self.cfg.ca1_threshold)
        burst_ne = burst_da * 0.8
        replayed: List[int] = []
        return HippoOut(ec=ec, dg=dg, ca3=ca3_full, ca1_out=recon,
                        mismatch=mismatch, vta_da_burst=burst_da,
                        lc_ne_burst=burst_ne, replayed=replayed)

    def store_episode(self, ec: torch.Tensor, dg: torch.Tensor,
                      ca3: torch.Tensor, error: float,
                      uncertainty: float) -> None:
        self.episodes.append((ec.detach().clone(), dg.detach().clone(),
                              ca3.detach().clone(), float(error),
                              float(uncertainty)))
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def swr_replay(self, cortical_write, learn: bool = True) -> List[int]:
        """Приоритетный реплей: P ∝ |err| + β·unc (§II.3.1).

        Возвращает индексы реплеированных эпизодов; каждый эпизод
        прокси-драйвит кортикальную консолидацию."""
        if not self.episodes:
            return []
        errs = torch.tensor([abs(e[3]) + self.cfg.replay_beta_unc * e[4]
                             for e in self.episodes])
        probs = (errs + 1e-6) / (errs + 1e-6).sum()
        k = min(self.cfg.n_replay_batch, len(self.episodes))
        idx = torch.multinomial(probs, k, replacement=False).tolist()
        for i in idx:
            ec, dg, ca3, err, unc = self.episodes[i]
            cortical_write(ec, err)     # прокси-градиент на кору
        return idx
