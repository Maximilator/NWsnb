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

from collections import deque
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


# ----------------------------------------------------------------------
def run_tinystories(data_path: str = "data/TinyStories_train_4k.txt",
                    preset: str = "tiny", max_stories: int = 24,
                    seq_len: int = 16, max_vocab: int = 128,
                    epochs: int = 2) -> Dict:
    """Реальные TinyStories → организм: perplexity до/после эпох.

    Токенизатор и словарь v9; каждый токен — кадр спайков в таламус;
    CE следующего токена → reward → DA/TD-пластичность + Adam-обучение
    ридаута. Между эпохами — сон с SWR-консолидацией."""
    from neuroweave9.data import (load_stories, tokenize_text,
                                  build_vocab_and_sequences)
    stories = load_stories(data_path, max_stories=max_stories)
    itos, stoi, x_seq, y_seq, _ = build_vocab_and_sequences(
        stories, max_vocab_size=max_vocab, seq_len=seq_len,
        batch_size=1)
    u = UnifiedOrganism(UnifiedConfig(vocab_size=len(itos)),
                        gcfg=genesis_cfg(preset))
    metrics: Dict = {"vocab": len(itos), "tokens": int(y_seq.numel())}
    ce_hist: List[float] = []
    for ep in range(epochs):
        ce_sum, n = 0.0, 0
        for b in range(x_seq.shape[1]):
            for t in range(x_seq.shape[0] - 1):
                tok = x_seq[t, b].argmax().unsqueeze(0)
                nxt = y_seq[t, b].unsqueeze(0)
                r = u.step_token(tok, nxt, learn=True)
                if u._pred_logits is not None and t > 0:
                    ce_sum += float(F.cross_entropy(
                        u._pred_logits.float(), y_seq[t, b:b+1]).item())
                    n += 1
        ce = ce_sum / max(1, n)
        ce_hist.append(ce)
        u.sleep(ticks=40)        # консолидация между эпохами
    metrics["ce_per_epoch"] = ce_hist
    metrics["perplexity_per_epoch"] = [
        float(torch.exp(torch.tensor(min(10.0, c)))) for c in ce_hist]
    metrics["organism"] = u.report()
    return metrics


# ----------------------------------------------------------------------
class RegenesisOrganism(UnifiedOrganism):
    """«True Bio-Organism»: моторная кора вместо MLP-декодера.

    Выходной токен — спайк WTA-победителя в пуле моторных нейронов
    (Закон 1). Афференты L5→моторный пул несут Tsodyks-Markram
    STP-депрессию: повтор нейрона истощает его везикулы — петля угасает
    без вето/штрафов (Закон 2). Обучение — трёхфакторное, без
    backprop; perplexity замеряется через softmax(V) чисто как метрика.
    """

    def __init__(self, cfg: Optional[UnifiedConfig] = None,
                 gcfg: Optional[GenesisConfig] = None, batch: int = 1,
                 seed: int = 0):
        super().__init__(cfg, gcfg, batch, seed)
        from ..motor import MotorPool
        z = self.gcfg.zones[-1]
        n_l5 = z.n_neurons - 2 * z.n_neurons // 3
        self.motor = MotorPool(self.ucfg.vocab_size, n_l5,
                               self.gcfg.hippo.n_ec, self.gcfg.input_dim,
                               self.gcfg.motor, seed=seed + 7)
        self.motor.attach_basal_ganglia(self.gcfg.bg.n_actions)
        self.motor.reset_state(batch)
        self._last_motor: Optional[Dict] = None
        # PFC-контур предсказания: w_pred — ожидание следующего токена
        # по предыдущему (предиктивное кодирование); масса пунктуации
        # в предсказании → «завершённость» → терпение DDM.
        v = self.ucfg.vocab_size
        self.w_pred = torch.randn(v, v) * 0.05
        self.punct_ids: set = set()
        self._recent_ids: deque = deque(maxlen=32)

    # ------------------------------------------------------------------
    def step_token(self, token_ids: torch.Tensor,
                   next_ids: Optional[torch.Tensor] = None,
                   learn: bool = True) -> Dict:
        """Токен → спайки → организм → моторный WTA → токен.

        reward: спайковое совпадение с next (биологическая оценка, не CE);
        обучение — err=H(target)−rate × пре-след × f_mod; пластичность
        заморожена при аденилатном долге (Закон 5)."""
        x = self._encode_frame(token_ids)
        # PFC-контур планирования: если предсказание не видит конца
        # фразы (пунктуация маловероятна и давно не было) — повысить
        # 5-HT-терпение: DDM не завершает эпизод преждевременно.
        prev_id = int(token_ids.flatten()[0].item())
        pred_p = torch.softmax(self.w_pred[prev_id].float(), dim=-1)
        punct_mass = float(pred_p[list(self.punct_ids)].sum()) \
            if self.punct_ids else 0.0
        recent_punct = any(i in self.punct_ids for i in self._recent_ids)
        patience = 0.0 if (recent_punct or punct_mass > 0.3) \
            else 0.6 * (1.0 - punct_mass)
        out = self.org.step(x, reward=0.0, learn=learn, drive=0.6,
                            patience=patience)
        zname = self.gcfg.zones[-1].name
        s_l5 = self.org._prev_l5.get(zname) \
            if hasattr(self.org, "_prev_l5") else torch.zeros(
                self.batch, self.motor.n_l5)
        hipp_v = self.org._prev_ca1
        lap = getattr(self.org, "_last_lap_slow", None)
        if lap is not None and lap.dim() == 1:
            lap = lap.unsqueeze(0)
        # Закон 1: канал BG-DDM открывает моторное действие
        self.motor.set_bg_decision(out.action)
        mout = self.motor.step(s_l5, hipp_v, lap, ou_on=learn)
        reward = 0.0
        if next_ids is not None and learn:
            tgt = int(next_ids[0].item())
            reward = 1.0 if mout.token == tgt else -0.5
            # PFC-модель ожидания: delta-правило по реальному next
            self.w_pred[prev_id] += 0.05 * (
                F.one_hot(torch.tensor(tgt), self.ucfg.vocab_size)
                .float() - pred_p)
            self.motor.learn(tgt, s_l5,
                             f_mod=float(self.org.chem.f_mod(True)),
                             plasticity_ok=not self.org.plasticity_blocked,
                             reward=reward)
            out2 = self.org.step(torch.zeros_like(x), reward=reward,
                                 learn=learn, drive=0.6)
        # Метрика: softmax(V) — только для perplexity, выбор идёт WTA
        logits = self.motor.v.mean(0).unsqueeze(0)
        self._pred_logits = logits
        self._last_motor = mout
        self._recent_ids.append(mout.token)
        return {"token": mout.token, "logits": logits, "margin": mout.margin,
                "stp_depleted": mout.stp_depleted, "reward": reward,
                "confidence": out.confidence, "sigma": out.sigma,
                "pathology": out.pathology}

    # ------------------------------------------------------------------
    def generate(self, n_tokens: int = 2000,
                 seed_tokens: Optional[List[int]] = None) -> Dict:
        """Свободная генерация: WTA-токен подаётся на вход следующего
        такта — организм сам ведёт последовательность (Закон 1).

        Диагностика: margin (разрыв V победителя), размер словаря в
        хвосте, длина максимальной петли повторов, STP-истощение."""
        ids: List[int] = list(seed_tokens or [0])
        margins: List[float] = []
        depletion: List[float] = []
        for _ in range(n_tokens):
            r = self.step_token(torch.tensor([ids[-1]]), learn=True)
            ids.append(r["token"])
            margins.append(r["margin"])
            depletion.append(r["stp_depleted"])
            if r["pathology"]:
                break
        tail = ids[-200:] if len(ids) >= 200 else ids
        # Максимальный прогон одного токена (петля)
        run, max_run = 1, 1
        for a, b in zip(ids, ids[1:]):
            run = run + 1 if a == b else 1
            max_run = max(max_run, run)
        return {"ids": ids, "margin_mean": float(
                    torch.tensor(margins).mean()) if margins else 0.0,
                "margin_max": float(max(margins)) if margins else 0.0,
                "tail_vocab": len(set(tail)),
                "total_vocab": len(set(ids)),
                "max_repeat_run": max_run,
                "stp_min": float(min(depletion)) if depletion else 1.0,
                "sigma": self.org.sigma,
                "pathology": self.org.pathology}


# ----------------------------------------------------------------------
# DevelopmentalController: поэтапное обучение (куррикулум)
# ----------------------------------------------------------------------
def _seq_pairs(story: str, stoi: Dict[str, int], seq_len: int,
               unk: int) -> List[int]:
    """История → список id по фиксированному словарю."""
    from neuroweave9.data import tokenize_text
    return [stoi.get(t, unk)
            for t in tokenize_text(story)[:seq_len + 1]]


def _checkpoint(u: "RegenesisOrganism", path: str,
                itos: List[str]) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"w_pred": u.w_pred.clone(),
            "itos": itos,
            "motor": {k: getattr(u.motor, k).clone()
                      for k in ("w_l5", "w_hip", "w_lap", "w_rec_m",
                                "stp_x", "stp_u", "out_fatigue",
                                "rate_aff", "rate_post")
                      if getattr(u.motor, k, None) is not None},
            "cols": [{"w_ff": c.w_ff.w.clone(),
                      "w_rec": c.w_rec.w.clone(),
                      "w_fb": c.w_fb.w.clone(),
                      "w_pv_in": c.w_pv_in.clone()}
                     for c in u.org.hier.cols]}
    torch.save(ckpt, path)


def load_checkpoint(u: "RegenesisOrganism", path: str) -> None:
    ckpt = torch.load(path, weights_only=False)
    for k, v in ckpt.get("motor", {}).items():
        setattr(u.motor, k, v)
    if "w_pred" in ckpt:
        u.w_pred = ckpt["w_pred"]
    for col, saved in zip(u.org.hier.cols, ckpt.get("cols", [])):
        col.w_ff.w = saved["w_ff"]
        col.w_rec.w = saved["w_rec"]
        col.w_fb.w = saved["w_fb"]
        col.w_pv_in = saved["w_pv_in"]


def run_curriculum(data_path: str = "data/10k-tinystories.parquet",
                   preset: str = "tiny", max_vocab: int = 128,
                   checkpoint_dir: str = "checkpoints",
                   stages: Optional[List[tuple]] = None,
                   only_stage: Optional[int] = None,
                   gen_tokens: int = 300,
                   sleep_ticks: int = 40) -> Dict:
    """Развитие организма по этапам: (n_stories, seq_len) на стадию.

    Между этапами — чекпоинт на диск и сон-консолидация; словарь
    строится один раз на первом этапе и фиксируется (нейроны моторного
    пула не меняют числа). Каждый этап измеряет CE→perplexity и хвост
    генерации: tail_vocab должен расти с обучением."""
    from neuroweave9.data import (load_stories, tokenize_text,
                                  build_vocab_and_sequences)
    stages = stages or [(500, 32), (4000, 64), (9800, 128)]
    # Словарь — на первом этапе (базовая лексика)
    first = load_stories(data_path, max_stories=stages[0][0])
    itos, stoi, _, _, _ = build_vocab_and_sequences(
        first, max_vocab_size=max_vocab, seq_len=stages[0][1],
        batch_size=1)
    u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                          gcfg=genesis_cfg(preset))
    u.punct_ids = {stoi[p] for p in (".", "!", "?") if p in stoi}
    unk = stoi.get("<unk>", 1)
    report: Dict = {"vocab": len(itos), "stages": []}
    stories_cache: List[str] = []
    for si, (n_stories, seq_len) in enumerate(stages):
        if only_stage is not None and si != only_stage:
            continue
        if len(stories_cache) < n_stories:
            stories_cache = load_stories(data_path,
                                         max_stories=n_stories)
        subset = stories_cache[:n_stories]
        ce_sum, n_tok = 0.0, 0
        for story in subset:
            ids = _seq_pairs(story, stoi, seq_len, unk)
            for a, b in zip(ids, ids[1:]):
                r = u.step_token(torch.tensor([a]),
                                 torch.tensor([b]), learn=True)
                if u._pred_logits is not None:
                    ce_sum += float(F.cross_entropy(
                        u._pred_logits.float(),
                        torch.tensor([b])).item())
                    n_tok += 1
        ce = ce_sum / max(1, n_tok)
        u.sleep(ticks=sleep_ticks)     # консолидация между этапами
        gen = u.generate(n_tokens=gen_tokens, seed_tokens=[1, 5, 9])
        ckpt_path = f"{checkpoint_dir}/curriculum_stage{si + 1}.pt"
        _checkpoint(u, ckpt_path, itos)
        report["stages"].append({
            "stage": si + 1, "stories": n_stories, "seq_len": seq_len,
            "tokens": n_tok, "ce": ce,
            "perplexity": float(torch.exp(torch.tensor(min(10.0, ce)))),
            "tail_vocab": gen["tail_vocab"],
            "max_repeat_run": gen["max_repeat_run"],
            "margin_mean": gen["margin_mean"],
            "sigma": gen["sigma"], "pathology": gen["pathology"],
            "checkpoint": ckpt_path})
    report["organism"] = u.report()
    return report
