"""
Train BiLSTM baseline and BiLSTM+MoE on CoNLL 2003 NER.
Saves checkpoints and prints per-epoch metrics.

Usage:
    python train.py --model moe   # train MoE model
    python train.py --model base  # train baseline
    python train.py               # train both and compare
"""

import argparse
import os
import sys
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

# Make src importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from dataset import CoNLL2003Dataset, build_vocab, NUM_LABELS, ID2LABEL
from model import BiLSTMNER, BiLSTMMoENER


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CFG = dict(
    embed_dim=128,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    num_experts=4,
    top_k=2,
    expert_hidden_dim=256,
    batch_size=32,
    lr=1e-3,
    epochs=15,
    max_len=128,
    min_freq=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir="checkpoints",
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    """Token-level accuracy and entity-level F1 (seqeval-style, simplified)."""
    preds = logits.argmax(dim=-1)  # (B, S)

    # Accuracy (ignore padding -100)
    active = mask.bool()
    correct = (preds[active] == labels[active]).sum().item()
    total = active.sum().item()
    acc = correct / total if total > 0 else 0.0

    # Entity-level F1 using seqeval if available, else skip
    try:
        from seqeval.metrics import f1_score, classification_report
        pred_tags, true_tags = [], []
        for b in range(labels.size(0)):
            p_seq, t_seq = [], []
            for s in range(labels.size(1)):
                if labels[b, s].item() == -100:
                    continue
                p_seq.append(ID2LABEL[preds[b, s].item()])
                t_seq.append(ID2LABEL[labels[b, s].item()])
            pred_tags.append(p_seq)
            true_tags.append(t_seq)
        f1 = f1_score(true_tags, pred_tags)
    except ImportError:
        f1 = float("nan")

    return acc, f1


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    for input_ids, labels, mask in loader:
        input_ids, labels, mask = input_ids.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, mask)              # (B, S, C)
        loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc, _ = compute_metrics(logits.detach(), labels, mask)
        total_loss += loss.item()
        total_acc += acc
        steps += 1

    return total_loss / steps, total_acc / steps


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_f1, steps = 0.0, 0.0, 0.0, 0
    for input_ids, labels, mask in loader:
        input_ids, labels, mask = input_ids.to(device), labels.to(device), mask.to(device)
        logits = model(input_ids, mask)
        loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
        acc, f1 = compute_metrics(logits, labels, mask)
        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1 if f1 == f1 else 0.0   # nan guard
        steps += 1

    return total_loss / steps, total_acc / steps, total_f1 / steps


def run(model_type: str, vocab: Dict, cfg: dict):
    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()}")
    print(f"  Device  : {device}")
    print(f"{'='*60}")

    # Data
    train_ds = CoNLL2003Dataset("train",      vocab, cfg["max_len"])
    val_ds   = CoNLL2003Dataset("validation", vocab, cfg["max_len"])
    test_ds  = CoNLL2003Dataset("test",       vocab, cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=train_ds.collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              collate_fn=val_ds.collate_fn,   num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              collate_fn=test_ds.collate_fn,  num_workers=2)

    # Model
    vocab_size = len(vocab)
    if model_type == "base":
        model = BiLSTMNER(
            vocab_size=vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_labels=NUM_LABELS,
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    else:
        model = BiLSTMMoENER(
            vocab_size=vocab_size,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_labels=NUM_LABELS,
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
            expert_hidden_dim=cfg["expert_hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(
            f"  Ep {epoch:02d}/{cfg['epochs']}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}  ({elapsed:.1f}s)"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test on best checkpoint
    model.load_state_dict(best_state)
    model = model.to(device)
    _, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\n  ✅ Test Accuracy : {test_acc:.4f}")
    print(f"  ✅ Test F1       : {test_f1:.4f}")

    # Save
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{model_type}_best.pt")
    torch.save({"model_state": best_state, "vocab": vocab, "cfg": cfg}, ckpt_path)
    print(f"  💾 Saved → {ckpt_path}")

    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "moe", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=CFG["lr"])
    parser.add_argument("--num_experts", type=int, default=CFG["num_experts"])
    parser.add_argument("--top_k", type=int, default=CFG["top_k"])
    args = parser.parse_args()

    CFG.update(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    print("Loading CoNLL 2003 and building vocabulary …")
    raw = load_dataset("conll2003", trust_remote_code=True)
    vocab = build_vocab(raw, min_freq=CFG["min_freq"])
    print(f"Vocabulary size: {len(vocab):,}")

    results = {}
    models_to_run = ["base", "moe"] if args.model == "both" else [args.model]
    for m in models_to_run:
        acc, f1 = run(m, vocab, CFG)
        results[m] = {"acc": acc, "f1": f1}

    if args.model == "both":
        print("\n" + "=" * 60)
        print("  FINAL COMPARISON")
        print("=" * 60)
        base_acc = results["base"]["acc"]
        moe_acc  = results["moe"]["acc"]
        base_f1  = results["base"]["f1"]
        moe_f1   = results["moe"]["f1"]
        print(f"  {'Model':<12} {'Accuracy':>10} {'F1':>10}")
        print(f"  {'-'*34}")
        print(f"  {'BiLSTM':<12} {base_acc:>10.4f} {base_f1:>10.4f}")
        print(f"  {'BiLSTM+MoE':<12} {moe_acc:>10.4f} {moe_f1:>10.4f}")
        print(f"  {'Δ (MoE-Base)':<12} {moe_acc-base_acc:>+10.4f} {moe_f1-base_f1:>+10.4f}")


if __name__ == "__main__":
    main()
