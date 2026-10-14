"""
Train BiLSTM baseline / dense-FFN control / BiLSTM+MoE on CoNLL 2003, over several seeds.
Entity F1 is computed once over the whole split with seqeval (not averaged over batches).

    python train.py --models none ffn moe --seeds 0 1 2 3 4
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import (CoNLL2003Dataset, ID2LABEL, NUM_LABELS, build_embedding_matrix,
                     build_vocabs, load_conll)
from model import NERTagger


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    gold, pred, correct, total = [], [], 0, 0
    for words, cases, chars, labels, mask in loader:
        words, cases, chars, mask = words.to(device), cases.to(device), chars.to(device), mask.to(device)
        p = model(words, cases, chars, mask).argmax(-1).cpu()
        for b in range(labels.size(0)):
            n = int(mask[b].sum())
            g = labels[b, :n].tolist(); q = p[b, :n].tolist()
            gold.append([ID2LABEL[i] for i in g]); pred.append([ID2LABEL[i] for i in q])
            correct += sum(int(a == c) for a, c in zip(g, q)); total += n
    return correct / total, f1_score(gold, pred)


def run(head, seed, data, args, device):
    set_seed(seed)
    emb, n_chars, train_ds, val_ds, test_ds = data
    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, collate_fn=ds.collate_fn)
    tr, va, te = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)
    model = NERTagger(emb, n_chars, NUM_LABELS, head=head, hidden_dim=args.hidden_dim,
                      dropout=args.dropout, num_experts=args.num_experts).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for words, cases, chars, labels, mask in tr:
            words, cases, chars, labels, mask = [t.to(device) for t in (words, cases, chars, labels, mask)]
            opt.zero_grad()
            loss = crit(model(words, cases, chars, mask).view(-1, NUM_LABELS), labels.view(-1))
            if head == "moe" and args.lb_coef > 0:
                loss = loss + args.lb_coef * model.block.aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item()
        v_acc, v_f1 = evaluate(model, va, device)
        sched.step(v_f1)
        print(f"  [{head} s{seed}] ep{ep:02d} loss={tot/len(tr):.4f} val_acc={v_acc:.4f} val_f1={v_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v_f1 > best_f1:
            best_f1, best_state = v_f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    t_acc, t_f1 = evaluate(model, te, device)
    usage = None
    if head == "moe":  # share of top-k routing assignments per expert on the test set
        model.eval(); acc_u, n = torch.zeros(args.num_experts), 0
        with torch.no_grad():
            for words, cases, chars, labels, mask in te:
                model(words.to(device), cases.to(device), chars.to(device), mask.to(device))
                acc_u += model.block.last_usage.cpu(); n += 1
        usage = [round(v, 3) for v in (acc_u / n).tolist()]
    print(f"==> {head} seed={seed} lb={args.lb_coef if head == 'moe' else '-'} params={n_params:,} val_f1={best_f1:.4f} test_acc={t_acc:.4f} test_f1={t_f1:.4f} usage={usage}", flush=True)
    return dict(head=head, seed=seed, lb_coef=args.lb_coef if head == "moe" else None, params=n_params,
                val_f1=best_f1, test_acc=t_acc, test_f1=t_f1, expert_usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["none", "ffn", "moe"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lb_coef", type=float, default=0.0, help="MoE load-balancing loss weight (0 = off)")
    ap.add_argument("--num_experts", type=int, default=4)
    ap.add_argument("--glove_dir", default="..")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    raw = load_conll()
    words = open(f"{args.glove_dir}/glove_words.txt").read().split("\n")
    vecs = np.load(f"{args.glove_dir}/glove_vecs.npy")
    w2i, c2i = build_vocabs(raw, words)
    emb, hit = build_embedding_matrix(w2i, words, vecs)
    print(f"word vocab {len(w2i):,} (GloVe hits {hit:,}), char vocab {len(c2i)}", flush=True)
    sp = lambda name, tr, wd=0.0: CoNLL2003Dataset(raw[name], w2i, c2i, train=tr, word_dropout=wd)
    data = (emb, len(c2i), sp("train", True, 0.05), sp("validation", False), sp("test", False))

    for seed in args.seeds:
        for head in args.models:
            r = run(head, seed, data, args, args.device)
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
