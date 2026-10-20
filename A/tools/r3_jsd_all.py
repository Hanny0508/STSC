"""四方法 JSD / PPL 同口径汇总（修复 tokenizer.pad_token 缺失）。

PPL：各方法编码器记录的值（ZTS 取操作点 JSON 的 perplexity；基线取 CSV 的 ppl 列）。
JSD：在覆盖语料上微调 gpt2-medium 得 baseline LM；在各自隐写语料上微调得 stego LM；
     两者在覆盖文本上的逐 token JS 散度均值。
"""

from __future__ import annotations

import importlib.util
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "legacy" / "zero-shot-GLS"
R3B = G / "tmp_saves" / "results" / "r3_baselines"
R3BR = G / "tmp_saves" / "results" / "r3_baselines_r3"
D3 = ROOT / "artifacts" / "d3"
TT = ROOT / "artifacts" / "table_x" / "text_table"

spec = importlib.util.spec_from_file_location("d3pj", str(ROOT / "tools" / "d3_ppl_jsd.py"))
pj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pj)


def read_csv(p: Path, col: str = "stegotext", limit: int | None = None):
    with p.open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get(col) or "").strip()]
    if limit:
        rows = rows[:limit]
    return rows


def ppl_from_csv(p: Path, limit: int = 400):
    vals = []
    for r in read_csv(p, limit=limit):
        try:
            vals.append(float(r.get("ppl") or "nan"))
        except ValueError:
            pass
    vals = [v for v in vals if v == v]
    return (round(st.mean(vals), 3), len(vals)) if vals else (None, 0)


# ---------------- 语料 ----------------
CORPORA: dict[str, list[str]] = {}
LIMIT = 400

for mb in (2, 3, 4):
    f = R3BR / f"rnn_maxbpw{mb}.csv"
    if f.is_file():
        CORPORA[f"RNN-Stega_mb{mb}"] = [r["stegotext"] for r in read_csv(f, limit=LIMIT)]

for mb in (2, 3, 4):
    f = R3BR / f"nls_mb{mb}_n900.csv"
    if f.is_file():
        CORPORA[f"NLS_mb{mb}"] = [r["stegotext"] for r in read_csv(f, limit=LIMIT)]

for mb in (2, 3, 4):
    f = R3BR / f"vae_mb{mb}_n900.csv"
    if f.is_file():
        CORPORA[f"VAE-Stega_mb{mb}"] = [r["stegotext"] for r in read_csv(f, limit=LIMIT)]

for tag, nm in (("P1", "ZTS_P1"), ("P2", "ZTS_P2"), ("P3", "ZTS_P3")):
    for cand in (D3 / f"{tag}_n900.json", D3 / f"{tag}.json"):
        if cand.is_file():
            d = json.loads(cand.read_text(encoding="utf-8"))
            CORPORA[nm] = [t for t in d.get("stegotext", []) if t and t.strip()][:LIMIT]
            break

print("corpora:", {k: len(v) for k, v in CORPORA.items()}, flush=True)

# ---------------- PPL ----------------
ppl = {}
for nm in CORPORA:
    if nm.startswith("ZTS_"):
        tag = nm.split("_")[1]
        for cand in (D3 / f"{tag}_n900.json", D3 / f"{tag}.json"):
            if cand.is_file():
                d = json.loads(cand.read_text(encoding="utf-8"))
                v = [x for x in (d.get("perplexity") or []) if x is not None]
                if v:
                    ppl[nm] = (round(st.mean(v), 3), len(v))
                break
    elif nm.startswith("RNN"):
        mb = nm[-1]
        ppl[nm] = ppl_from_csv(R3BR / f"rnn_maxbpw{mb}.csv")
    elif nm.startswith("VAE"):
        mb = nm[-1]
        ppl[nm] = ppl_from_csv(R3BR / f"vae_mb{mb}_n900.csv")
    elif nm.startswith("NLS"):
        mb = nm[-1]
        ppl[nm] = ppl_from_csv(R3BR / f"nls_mb{mb}_n900.csv")
print("ppl:", ppl, flush=True)

# ---------------- JSD ----------------
from transformers import GPT2Tokenizer

tok = GPT2Tokenizer.from_pretrained("gpt2-medium")
tok.pad_token = tok.eos_token          # ★ 关键修复

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
covers = [t for t in json.loads((D3 / "plain_controls.json").read_text(encoding="utf-8"))["texts"]
          if t and t.strip()][:LIMIT]

cover_dir = D3 / "lm_cover_jsd"
if not (cover_dir / "config.json").is_file():
    print("finetune cover LM ...", flush=True)
    pj.finetune(covers, cover_dir, epochs=5)
mcover = pj.load_lm(cover_dir, dev)

jsd = {}
for name, texts in CORPORA.items():
    texts = [t for t in texts if t and t.strip()][:LIMIT]
    if len(texts) < 50:
        print(f"  跳过 {name}（样本太少 {len(texts)}）", flush=True)
        continue
    sd = D3 / f"lm_jsd_{name}"
    try:
        if not (sd / "config.json").is_file():
            print(f"finetune {name} ...", flush=True)
            pj.finetune(texts, sd, epochs=5)
        ms = pj.load_lm(sd, dev)
        val = pj.jsd_between(ms, mcover, covers, tok, dev)
        jsd[name] = round(val, 4)
        print(f"  {name}: JSD={val:.4f}  n={len(texts)}", flush=True)
        del ms
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {name} 失败: {type(exc).__name__}: {str(exc)[:160]}", flush=True)

out = TT / "results" / "jsd_ppl_all_methods.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schema": "zsc-tsd.jsd-ppl-all-methods.v1",
    "limit_per_corpus": LIMIT,
    "cover": "artifacts/d3/plain_controls.json[:400]",
    "jsd_model": "gpt2-medium (finetuned 5 epochs, lr 3e-5)",
    "ppl_source": "ZTS: operating-point JSON perplexity (Llama-2-7B); baselines: CSV ppl column",
    "jsd": jsd, "ppl": {k: v[0] for k, v in ppl.items()}, "ppl_n": {k: v[1] for k, v in ppl.items()},
}, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, flush=True)
