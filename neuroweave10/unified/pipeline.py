"""UnifiedOrganism: v9-энкодер → v10-организм → v9-ридаут → CE→reward.

Биологическая эффективность: входной токен — один спайк-кадр в
таламический FO-релей (латентность несёт салиентность — редкие токены
спайкуют раньше, Aston-Jones); внутри дендритного дерева драйв идёт по
sparse_drive (v8): только активные пресинаптические входы платят
матричный драйв — событийная экономия как в мозге.

Синергия:
  * v8 sparse ops → экономия вычислений (EventStats считает долю работы)
  * v9 encoding+readout → языковая задача (TinyStories next-token)
  * v10 организм → кортекс/память/сознание/химия — обучение трёхфакторным
    правилом с f_mod-гейтом (DA+NE+eCB+CORT)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from neuroweave8.core.sparse import sparse_drive, EventStats
from neuroweave9.encoding import TemporalPhaseEncoder
from neuroweave9.readout import LaplaceContextReadout

from ..config import GenesisConfig, genesis_cfg
from ..organism.genesis import GenesisOrganism, GenesisOutput


@dataclass
class UnifiedConfig:
    preset: str = "tiny"
    vocab_size: int = 128
    burst_length: int = 5
    max_latency: int = 8
    enc_mode: str = "latency"      # салиентность входа встроена
    reward_scale: float = 2.0      # CE → reward амплитуда


@dataclass
class UnifiedReport:
    tokens: int
    ce: float
    perplexity: float
    sigma: float
    confidence: float
    m_ratio: float
    sparse_saving: float
    pathology: Optional[str]


class UnifiedOrganism:
    """Единый организм v8+v9+v10."""

    def __init__(self, cfg: Optional[UnifiedConfig] = None,
                 gcfg: Optional[GenesisConfig] = None, batch: int = 1,
                 seed: int = 0):
        self.ucfg = cfg or UnifiedConfig()
        self.gcfg = gcfg or genesis_cfg(self.ucfg.preset)
        self.batch = batch
        self.org = GenesisOrganism(self.gcfg, batch=batch, seed=seed)
        self.org.reset_state(batch)
        self.encoder = TemporalPhaseEncoder(
            self.ucfg.vocab_size, self.ucfg.burst_length,
            self.ucfg.max_latency, self.ucfg.enc_mode)
        # Проекция vocab→input_dim: разреженная (k-WTA) — экономия
        self.w_in = torch.randn(self.gcfg.input_dim,
                                self.ucfg.vocab_size) * 0.4
        self.readout = LaplaceContextReadout(
            cortical_dim=self.gcfg.zones[-1].n_neurons,
            laplace_dim=self.gcfg.gw.d_gw,
            vocab_size=self.ucfg.vocab_size, hidden=64, lr=2e-3)
        self.event_stats = EventStats()
        self._pred_logits: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _encode_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        """Токены [B] → вход организма [B,input_dim]: latency-спайк ×
        салиентность, разреженная проекция на input_dim."""
        enc = self.encoder.encode_latency(tokens.unsqueeze(-1))
        frame = enc[:, :, :]                     # [T,B,V]
        sal = frame.max(0).values                # [B,V] салиентность
        x = sparse_drive(sal, self.w_in,
                         out_idx=torch.arange(self.gcfg.input_dim))
        self.event_stats.record(self.ucfg.vocab_size,
                                int((sal > 0).sum().item()),
                                self.gcfg.input_dim, self.batch)
        return x * 3.0

    def _readout_logits(self, out: GenesisOutput) -> torch.Tensor:
        """PFC-спайки + GW-бродкаст → логиты словаря."""
        pfc = out.spikes.get(self.gcfg.zones[-1].name) \
            if hasattr(out, "spikes") and isinstance(
                getattr(out, "spikes", None), dict) else None
        if pfc is None:
            pfc = self.org._prev_pfc if hasattr(self.org, "_prev_pfc") \
                else torch.zeros(self.batch, self.gcfg.zones[-1].n_neurons)
        gw = out.gw_content
        gw_t = torch.stack([g.detach() if torch.is_tensor(g)
                            else torch.as_tensor(g, dtype=torch.float32)
                            for g in gw]) if isinstance(gw, list) else gw.detach()
        gw_flat = gw_t.flatten().float()[:self.gcfg.gw.d_gw]
        gw_flat = F.pad(gw_flat, (0, max(0, self.gcfg.gw.d_gw
                                       - gw_flat.numel()))).unsqueeze(0)
        zl = torch.zeros(self.batch, self.ucfg.vocab_size)
        return self.readout(pfc.float(), gw_flat, zl)

    # ------------------------------------------------------------------
    def step_token(self, token_ids: torch.Tensor,
                   next_ids: Optional[torch.Tensor] = None,
                   learn: bool = True) -> Dict:
        """Токены [B] текущего шага → предсказание; если next_ids —
        обучение по CE-ошибке (reward = -CE·scale → DA-канал)."""
        x = self._encode_frame(token_ids)
        reward = 0.0
        if self._pred_logits is not None and next_ids is not None:
            # Отрицательная CE → reward: ошибка предсказания драйвит
            # BG-TD и f_mod-пластичность (правило 3-х факторов).
            ce = F.cross_entropy(self._pred_logits.float(),
                                 next_ids.long())
            reward = float(-ce.item() * self.ucfg.reward_scale)
            if learn:
                self.readout.optimizer.zero_grad()
                ce.backward()
                self.readout.optimizer.step()
        out = self.org.step(x, reward=reward, learn=learn, drive=0.6)
        logits = self._readout_logits(out)
        self._pred_logits = logits
        return {"logits": logits, "reward": reward,
                "confidence": out.confidence, "sigma": out.sigma,
                "action": out.action, "pathology": out.pathology,
                "gw": out.gw_content}

    # ------------------------------------------------------------------
    def run_sequence(self, seq: torch.Tensor, learn: bool = True) -> UnifiedReport:
        """seq [T] → обучающий проход next-token; отчёт о точности и здоровье."""
        ce_sum, n = 0.0, 0
        for t in range(seq.numel() - 1):
            r = self.step_token(seq[t:t + 1], seq[t + 1:t + 2], learn=learn)
            if self._pred_logits is not None and t > 0:
                ce_sum += float(F.cross_entropy(
                    self._pred_logits.float(), seq[t:t + 1].long()).item())
                n += 1
            if r["pathology"]:
                break
        ce = ce_sum / max(1, n)
        return UnifiedReport(
            tokens=int(seq.numel()), ce=ce,
            perplexity=float(torch.exp(torch.tensor(min(8.0, ce)))),
            sigma=self.org.sigma, confidence=float(self.org._last_meta.confidence
                                                   if self.org._last_meta else 0.0),
            m_ratio=float(getattr(self.org._last_meta, 'm_ratio', 0.0) if getattr(self.org, '_last_meta', None) is not None else 0.0),
            sparse_saving=self.event_stats.saving,
            pathology=self.org.pathology)

    def sleep(self, ticks: int = 150) -> Dict:
        return self.org.sleep_cycle(ticks)

    def report(self) -> Dict:
        return {"sigma": self.org.sigma,
                "m_ratio": float(getattr(self.org._last_meta, 'm_ratio', 0.0) if getattr(self.org, '_last_meta', None) is not None else 0.0),
                "dash": getattr(self.org, "_last_dash", {}),
                "sparse_saving": self.event_stats.saving,
                "episodes": len(self.org.hipp.episodes),
                "pathology": self.org.pathology}
