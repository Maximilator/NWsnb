"""Этап 2 куррикулума: 4000 историй seq_len=64 → чекпоинт → генерация 1000.

Логирует: RSS-память каждые 400 историй, средний reward окнами,
метрики генерации. Выводит JSON-итог в конец.
"""
import sys
sys.path.insert(0, "/home/ubuntu/repos/NWsnb")
import json
import os
import resource

import torch
import torch.nn.functional as F

from neuroweave10.unified.pipeline import (
    RegenesisOrganism, UnifiedConfig, _seq_pairs, _checkpoint)
from neuroweave10.config import genesis_cfg
from neuroweave9.data import load_stories, build_vocab_and_sequences

DATA = "data/10k-tinystories.parquet"
N_STORIES, SEQ_LEN, MAX_VOCAB = 4000, 64, 128
CKPT = "checkpoints/curriculum_stage2.pt"

stories = load_stories(DATA, max_stories=N_STORIES)
itos, stoi, *_ = build_vocab_and_sequences(
    stories[:500], max_vocab_size=MAX_VOCAB, seq_len=SEQ_LEN,
    batch_size=1)
u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                      gcfg=genesis_cfg("tiny"))
u.punct_ids = {stoi[p] for p in (".", "!", "?") if p in stoi}
unk = stoi.get("<unk>", 1)
prev_ckpt = "checkpoints/curriculum_stage1.pt"
if os.path.exists(prev_ckpt):
    from neuroweave10.unified.pipeline import load_checkpoint
    load_checkpoint(u, prev_ckpt)
    print(f"[stage2] resumed from {prev_ckpt}", flush=True)

def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

ce_sum, n_tok, rewards = 0.0, 0, []
for i, story in enumerate(stories):
    ids = _seq_pairs(story, stoi, SEQ_LEN, unk)
    for a, b in zip(ids, ids[1:]):
        r = u.step_token(torch.tensor([a]), torch.tensor([b]),
                         learn=True)
        rewards.append(r["reward"])
        if u._pred_logits is not None:
            ce_sum += float(F.cross_entropy(
                u._pred_logits.float(), torch.tensor([b])).item())
            n_tok += 1
    if (i + 1) % 400 == 0:
        wr = sum(rewards[-12000:]) / max(1, len(rewards[-12000:]))
        print(f"[stage2] stories={i+1} tok={n_tok} rss={rss_mb():.0f}MB "
              f"ce={ce_sum/max(1,n_tok):.3f} reward_ema={wr:.4f} "
              f"sigma={u.org.sigma:.3f} patho={u.org.pathology}",
              flush=True)

u.sleep(ticks=40)
_checkpoint(u, CKPT, itos)

# Генерация 1000 токенов + текст
gen = u.generate(n_tokens=1000, seed_tokens=[1, 5, 9])
ids = gen["ids"]
text = " ".join(itos[i] if i < len(itos) else "<unk>" for i in ids)
reward_mean = sum(rewards) / max(1, len(rewards))
reward_tail = sum(rewards[-16000:]) / max(1, len(rewards[-16000:]))
out = {
    "tokens_trained": n_tok,
    "ce": ce_sum / max(1, n_tok),
    "reward_mean": reward_mean,
    "reward_tail": reward_tail,
    "reward_first2k": sum(rewards[:2000]) / max(1, len(rewards[:2000])),
    "reward_last2k": sum(rewards[-2000:]) / max(1, len(rewards[-2000:])),
    "rss_mb": rss_mb(),
    "gen": {k: gen[k] for k in ("margin_mean", "margin_max",
                                "tail_vocab", "total_vocab",
                                "max_repeat_run", "stp_min",
                                "sigma", "pathology")},
    "text_first200": " ".join((itos[i] if i < len(itos) else "<unk>")
                             for i in ids[:200]),
    "text_last200": " ".join((itos[i] if i < len(itos) else "<unk>")
                            for i in ids[-200:]),
}
print("RESULT_JSON:" + json.dumps(out), flush=True)
