"""PPL and JSD for the text-domain table.

Follows the original evaluation pipeline (``evaluate/GPT/train_gpt.py`` and
``evaluate/GPT/jsd.py``) but without the deprecated ``datasets``/``Trainer``
API:

  * fine-tune gpt2-medium on a cover corpus  -> baseline language model
  * fine-tune gpt2-medium on the stego corpus -> stego language model
  * JSD = mean over tokens of the Jensen-Shannon divergence between the two
    models' next-token distributions, evaluated on the cover texts

PPL is reported separately: the encoder already records the perplexity of each
stego text under the Llama generator, and this script averages whatever
``perplexity`` fields it finds in the supplied result files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

MODEL_NAME = "gpt2-medium"


def load_texts(path: Path, key: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(t) for t in payload]
    if key in payload:
        return [str(t) for t in payload[key]]
    raise KeyError(f"{path} has no '{key}' list")


def finetune(texts, out_dir: Path, *, epochs=5, lr=3e-5, batch=16,
             max_len=64, seed=42) -> Path:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.train()

    # Some control corpora contain empty strings (a few prompts yield no text);
    # GPT-2's tokenizer returns a zero-length tensor for them, which breaks the
    # batch concatenation below. Drop them before tokenising.
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    encodings = [
        tokenizer(t, truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="pt")
        for t in texts
    ]
    encodings = [e for e in encodings if e["input_ids"].numel() > 0]
    if not encodings:
        raise SystemExit("no usable texts for fine-tuning")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    order = list(range(len(encodings)))

    for epoch in range(epochs):
        random.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            ids = torch.cat([encodings[i]["input_ids"] for i in chunk]).to(device)
            mask = torch.cat([encodings[i]["attention_mask"] for i in chunk]).to(device)
            labels = ids.clone()
            labels[mask == 0] = -100
            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            total += float(loss) * len(chunk)
        print(f"    epoch {epoch + 1}/{epochs}  loss {total / len(order):.4f}",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def load_lm(directory: Path, device):
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(directory).to(device).eval()
    return model


def jsd_between(model1, model2, texts, tokenizer, device, *, max_len=64,
                batch=8) -> float:
    """Mean per-token Jensen-Shannon divergence between two LMs."""

    total, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            enc = tokenizer(chunk, truncation=True, max_length=max_len,
                            padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            logits1 = model1(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
            logits2 = model2(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
            p1 = torch.log_softmax(logits1.float(), dim=-1)
            p2 = torch.log_softmax(logits2.float(), dim=-1)
            m = torch.logsumexp(
                torch.stack([p1, p2], dim=0), dim=0
            ) - math.log(2.0)

            kl1 = (p1.exp() * (p1 - m)).sum(dim=-1)
            kl2 = (p2.exp() * (p2 - m)).sum(dim=-1)
            jsd = 0.5 * (kl1 + kl2) * mask[:, 1:].float()
            total += float(jsd.sum())
            tokens += int(mask[:, 1:].sum())
    return total / max(tokens, 1)


def average_perplexity(paths: list[Path]) -> float | None:
    values: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per = payload.get("perplexity")
        if isinstance(per, list):
            values.extend(float(v) for v in per if v is not None)
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cover-json", type=Path,
                        default=Path("artifacts/d3/plain_controls.json"))
    parser.add_argument("--cover-key", default="texts")
    parser.add_argument("--points", default="opB,opC")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("artifacts/d3/ppl_jsd.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    cover_texts = load_texts(root / args.cover_json, args.cover_key)[: args.limit]
    print(f"cover texts: {len(cover_texts)}")

    cover_model_dir = root / "artifacts" / "d3" / "lm_cover"
    if (cover_model_dir / "config.json").is_file():
        print("cover language model already trained")
    else:
        print("fine-tuning the cover language model ...")
        finetune(cover_texts, cover_model_dir, epochs=args.epochs)
    model_cover = load_lm(cover_model_dir, device)

    results: dict[str, dict] = {}
    for name in [p for p in args.points.split(",") if p]:
        point_path = root / "artifacts" / "d3" / f"{name}.json"
        if not point_path.is_file():
            print(f"{name}: missing, skipped")
            continue
        stego_texts = [t for t in load_texts(point_path, "stegotext")
                       if t and t.strip()][: args.limit]
        print(f"\n{name}: {len(stego_texts)} stego texts")

        stego_model_dir = root / "artifacts" / "d3" / f"lm_{name}"
        if not (stego_model_dir / "config.json").is_file():
            finetune(stego_texts, stego_model_dir, epochs=args.epochs)
        model_stego = load_lm(stego_model_dir, device)

        jsd = jsd_between(model_stego, model_cover, cover_texts, tokenizer, device)
        ppl = average_perplexity([point_path])
        results[name] = {"jsd": round(jsd, 4), "ppl": None if ppl is None else round(ppl, 3)}
        print(f"  {name}: JSD={jsd:.4f}  PPL={results[name]['ppl']}")
        del model_stego
        torch.cuda.empty_cache()

    out = (root / args.out) if not args.out.is_absolute() else args.out
    out.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
