"""Text-domain detectability under the D3 (transcription-level) adversary.

Reproduces the three steganalyzers named in the manuscript -- BERT-cls,
LS-CNN and TS-BiRNN -- with a self-contained data pipeline, because the
original code depends on the removed ``torchtext.data`` API.

Conditions
----------
clean
    cover text                       vs  steganographic text
transcript
    ASR transcript of cover audio    vs  ASR transcript of stego audio

Both conditions use balanced classes (p = 0.5), so the reported triple must
satisfy ``Acc = 1 - 0.5*FPR - 0.5*FNR`` exactly; the script asserts this for
every cell.  Table X of the previous submission violates the same identity,
which is why it had to be withdrawn.

Usage
-----
    python tools/d3_text_detectability.py \
        --inventory artifacts/d3/dataset_inventory.json \
        --covers    artifacts/d3/covers.json \
        --out       artifacts/d3/d3_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np

# Point the transformers cache at the project's own copy before transformers is
# imported, so bert-medium loads offline and the tokenizer can fall back to the
# mirror when the primary hub is unreachable.
_ROOT_HINT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "HF_HOME", str(_ROOT_HINT / "legacy" / "zero-shot-GLS" / "tmp_saves" / "hg_cache")
)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WORD_RE = re.compile(r"[a-z0-9']+")

FIX_LEN = 40          # matches data.Field(fix_length=40) in the original code
SEED = __import__('os').environ.get('ZSC_D3_SEED', '2024')
SEED = int(SEED)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def word_tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


class WordVocab:
    def __init__(self, min_count: int = 1, max_size: int = 30000):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.vectors: np.ndarray | None = None
        self._min_count = min_count
        self._max_size = max_size

    def build(self, texts: list[str]) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in word_tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, count in ordered:
            if count < self._min_count:
                continue
            if len(self.stoi) >= self._max_size:
                break
            self.stoi[token] = len(self.stoi)

    def load_glove(self, path: Path, dim: int = 300) -> int:
        vectors = np.zeros((len(self.stoi), dim), dtype=np.float32)
        found = 0
        wanted = {token: idx for token, idx in self.stoi.items() if idx > 1}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip().split(" ")
                if len(parts) != dim + 1:
                    continue
                token = parts[0]
                idx = wanted.get(token)
                if idx is None:
                    continue
                vectors[idx] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
        self.vectors = vectors
        return found

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(tok, 1) for tok in word_tokenize(text)][:FIX_LEN]
        if len(ids) < FIX_LEN:
            ids += [0] * (FIX_LEN - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.stoi)


class WordDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: WordVocab):
        self.ids = torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.ids[index], self.labels[index]


class BertTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = [
            tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        enc = self.encodings[index]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": self.labels[index],
        }


# --------------------------------------------------------------------------
# models (mirroring the originals)
# --------------------------------------------------------------------------
class BertCls(nn.Module):
    """Mirrors Bert_cls/Bert_cls.py (prajjwal1/bert-medium + one linear head)."""

    def __init__(self, freeze_first: int = 2):
        super().__init__()
        from transformers import AutoModel

        self.bert_model = AutoModel.from_pretrained("prajjwal1/bert-medium")
        self.bert_model.encoder.layer[0:freeze_first].requires_grad_(False)
        self.mlp = nn.Linear(512, 2)

    @property
    def device(self):
        return self.mlp.weight.device

    def forward(self, input_ids, attention_mask):
        out = self.bert_model(input_ids, attention_mask=attention_mask)
        return self.mlp(out.pooler_output)


class LSCnn(nn.Module):
    """Mirrors LS_CNN/LS_CNN.py."""

    def __init__(self, vocab_size, embed_dim=300, class_num=2, kernel_num=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5, frozen_vectors=None):
        super().__init__()
        self.embed_A = nn.Embedding(vocab_size, embed_dim)
        self.embed_B = nn.Embedding(vocab_size, embed_dim)
        if frozen_vectors is not None:
            self.embed_B.weight.data.copy_(torch.from_numpy(frozen_vectors))
        self.embed_B.weight.requires_grad = False
        self.conv_embed = nn.Conv2d(2, 1, (1, 1))
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, embed_dim)) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)

    def forward(self, x):
        x_a = self.embed_A(x)
        x_b = self.embed_B(x)
        x = torch.cat([x_a.unsqueeze(3), x_b.unsqueeze(3)], 3).permute(0, 3, 1, 2)
        x = self.conv_embed(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return self.fc1(self.dropout(x))


class TSBiRNN(nn.Module):
    """Mirrors TS-BiRNN/TS_BiRNN.py."""

    def __init__(self, vocab_size, embed_dim=300, hidden=128, layers=2,
                 class_num=2, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(2 * hidden, class_num)

    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        return self.fc1(out[:, -1, :])


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------
def evaluate(model, loader, device, *, bert: bool, collect: bool = False) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    per_sample: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            if bert:
                labels = batch["label"].to(device)
                logits = model(batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device))
            else:
                ids, labels = batch
                logits = model(ids.to(device))
                labels = labels.to(device)
            pred = logits.argmax(dim=1)
            if collect:
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
                for label_i, prob_i, pred_i in zip(
                    labels.tolist(), probs.tolist(), pred.tolist()
                ):
                    per_sample.append(
                        {
                            "label": int(label_i),
                            "prob_stego": round(float(prob_i), 6),
                            "predicted": int(pred_i),
                        }
                    )
            tp += int(((pred == 1) & (labels == 1)).sum())
            fp += int(((pred == 1) & (labels == 0)).sum())
            tn += int(((pred == 0) & (labels == 0)).sum())
            fn += int(((pred == 0) & (labels == 1)).sum())

    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    result = {
        "n": total,
        "acc": round(100 * acc, 2),
        "fpr": round(100 * fpr, 2),
        "fnr": round(100 * fnr, 2),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "identity_residual": round(acc - (1 - 0.5 * fpr - 0.5 * fnr), 6),
        "identity_holds": abs(acc - (1 - 0.5 * fpr - 0.5 * fnr)) < 0.005,
    }
    if collect:
        result["_per_sample"] = per_sample
    return result


def train_bert(model, loader, device, epochs, lr, bert=True):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device))
            F.cross_entropy(logits, labels).backward()
            optimizer.step()


def train_word_model(model, loader, device, epochs, lr):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    for _ in range(epochs):
        model.train()
        for ids, labels in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(ids.to(device)), labels.to(device)).backward()
            optimizer.step()


# --------------------------------------------------------------------------
def split_indices(n: int, test_fraction: float = 0.25) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(SEED).shuffle(indices)
    cut = int(round(n * (1 - test_fraction)))
    return sorted(indices[:cut]), sorted(indices[cut:])


def _attach_predictions(result, covers, stegos, test_idx, enabled):
    # r3_predictions_applied
    if not enabled:
        return result
    raw = result.pop("_per_sample", [])
    n_test = len(test_idx)
    rows = []
    for position, row in enumerate(raw):
        if position < n_test:
            source = test_idx[position]
            text = covers[source]
            role = "cover"
        else:
            source = test_idx[position - n_test]
            text = stegos[source]
            role = "stego"
        rows.append(
            {
                "position": position,
                "role": role,
                "source_index": int(source),
                "label": row["label"],
                "prob_stego": row["prob_stego"],
                "predicted": row["predicted"],
                "correct": int(row["label"] == row["predicted"]),
                "text": text,
            }
        )
    result["_predictions"] = rows
    return result

def _write_predictions(base, condition: str, detector: str, rows: list,
                       root) -> None:
    directory = base if base.is_absolute() else (root / base)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{condition}_{detector}.predictions.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1) + "\\n",
        encoding="utf-8",
    )
    with (directory / f"{condition}_{detector}.predictions.csv").open(
        "w", newline="", encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["position", "role", "source_index", "label",
                        "prob_stego", "predicted", "correct", "text"],
        )
        writer.writeheader()
        writer.writerows(rows)

def run_condition(name, covers, stegos, *, detector, device, epochs, glove,
                  limit, bert_tokenizer=None, collect_predictions=False):
    n = min(len(covers), len(stegos), limit)
    covers, stegos = covers[:n], stegos[:n]

    train_idx, test_idx = split_indices(n)

    def gather(pool, idx):
        return [pool[i] for i in idx]

    train_texts = gather(covers, train_idx) + gather(stegos, train_idx)
    train_labels = [0] * len(train_idx) + [1] * len(train_idx)
    test_texts = gather(covers, test_idx) + gather(stegos, test_idx)
    test_labels = [0] * len(test_idx) + [1] * len(test_idx)

    if detector == "bert":
        train_ds = BertTextDataset(train_texts, train_labels, bert_tokenizer)
        test_ds = BertTextDataset(test_texts, test_labels, bert_tokenizer)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        model = BertCls().to(device)
        train_bert(model, train_loader, device, epochs, 2e-4)
        return _attach_predictions(
            evaluate(model, test_loader, device, bert=True,
                     collect=collect_predictions),
            covers, stegos, test_idx, collect_predictions,
        )

    vocab = WordVocab()
    vocab.build(train_texts)
    glove_hits = 0
    if glove is not None and glove.is_file():
        glove_hits = vocab.load_glove(glove)

    train_ds = WordDataset(train_texts, train_labels, vocab)
    test_ds = WordDataset(test_texts, test_labels, vocab)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    if detector == "lscnn":
        model = LSCnn(len(vocab), frozen_vectors=vocab.vectors).to(device)
    else:
        model = TSBiRNN(len(vocab)).to(device)
    train_word_model(model, train_loader, device, epochs, 1e-3)

    result = _attach_predictions(
        evaluate(model, test_loader, device, bert=False,
                 collect=collect_predictions),
        covers, stegos, test_idx, collect_predictions,
    )
    result["vocab_size"] = len(vocab)
    result["glove_hits"] = glove_hits
    result["glove_used"] = bool(glove_hits)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--inventory", type=Path,
                        default=Path("artifacts/d3/dataset_inventory.json"))
    parser.add_argument("--covers", type=Path, default=Path("artifacts/d3/covers.json"))
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("artifacts/d3/plain_controls.json"),
        help="Unembedded gener-ated texts aligned index-by-index with the stego "
        "texts; used by the 'control' condition to isolate the embedding.",
    )
    parser.add_argument("--glove", type=Path, default=None)
    parser.add_argument(
        "--operating-point",
        type=Path,
        default=None,
        help="JSON produced by d3_generate_operating_point.py; adds an "
        "'op' condition comparing its unembedded control with its stego text.",
    )
    parser.add_argument("--detectors", default="bert,lscnn,tsbirnn")
    parser.add_argument("--conditions", default="clean,control,transcript")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        help="directory for per-sample detector predictions",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    inventory = json.loads((root / args.inventory).read_text(encoding="utf-8"))["records"]
    stego_text = [r["stegotext"] for r in inventory if r.get("stegotext")]
    stego_transcript = [r["transcript"] for r in inventory
                        if r.get("stegotext") and r.get("transcript")]
    print(f"stego texts      : {len(stego_text)}")
    print(f"stego transcripts: {len(stego_transcript)}")

    covers = json.loads((root / args.covers).read_text(encoding="utf-8"))
    cover_text = covers["texts"]
    cover_transcript = covers.get("transcripts") or []
    print(f"cover texts      : {len(cover_text)}")
    print(f"cover transcripts: {len(cover_transcript)}")

    control_text: list[str] = []
    if (root / args.controls).is_file():
        control_payload = json.loads(
            (root / args.controls).read_text(encoding="utf-8")
        )
        control_text = [
            t for t in control_payload["texts"] if isinstance(t, str) and t.strip()
        ]
        print(f"control texts    : {len(control_text)}  "
              f"(unembedded generations paired with the stego texts)")

    op_stego: list[str] = []
    op_control: list[str] = []
    if args.operating_point is not None:
        op_path = (
            args.operating_point
            if args.operating_point.is_absolute()
            else (root / args.operating_point)
        )
        if op_path.is_file():
            payload = json.loads(op_path.read_text(encoding="utf-8"))
            op_stego = [t for t in payload["stegotext"] if t and t.strip()]
            op_control = [t for t in payload["control"] if t and t.strip()]
            print(f"operating point  : {payload['params']}")
            print(f"  paired samples : {min(len(op_stego), len(op_control))}")

    glove = args.glove
    if glove is not None and not glove.is_absolute():
        glove = root / glove

    results: dict[str, dict] = {}
    for condition in args.conditions.split(","):
        if condition == "clean":
            pool_c, pool_s = cover_text, stego_text
        elif condition == "control":
            pool_c, pool_s = control_text, stego_text
        elif condition == "op":
            pool_c, pool_s = op_control, op_stego
        else:
            pool_c, pool_s = cover_transcript, stego_transcript
        if not pool_c or not pool_s:
            print(f"skipping condition {condition}: missing data")
            continue

        results[condition] = {}
        for detector in args.detectors.split(","):
            epochs = args.epochs
            if epochs is None:
                # Original training budgets: Bert_cls used 10 epochs with
                # lr 2e-4; LS_CNN and TS-BiRNN used 20 epochs with lr 1e-3.
                epochs = 10 if detector == "bert" else 20
            print(f"\n=== condition={condition}  detector={detector}  epochs={epochs}")
            tokenizer = None
            if detector == "bert":
                from transformers import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            result = run_condition(
                condition, pool_c, pool_s, detector=detector, device=device,
                epochs=epochs, glove=glove, limit=args.limit,
                bert_tokenizer=tokenizer,
                collect_predictions=args.predictions_out is not None,
            )
            if args.predictions_out is not None:
                _write_predictions(
                    args.predictions_out, condition, detector,
                    result.pop("_predictions", []), root,
                )
            print("   ", json.dumps(result, ensure_ascii=False))
            results[condition][detector] = result

    out = (root / args.out) if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "seed": SEED,
                    "fix_len": FIX_LEN,
                    "limit": args.limit,
                    "class_balance": 0.5,
                    "identity": "Acc = 1 - 0.5*FPR - 0.5*FNR",
                },
                "counts": {k: (len(v) if isinstance(v, list) else v)
                           for k, v in (("stego_text", stego_text),
                                        ("stego_transcript", stego_transcript),
                                        ("cover_text", cover_text),
                                        ("cover_transcript", cover_transcript))},
                "results": results,
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")

    all_ok = all(
        cell["identity_holds"]
        for cond in results.values()
        for cell in cond.values()
    )
    print(f"identity check on every cell: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
