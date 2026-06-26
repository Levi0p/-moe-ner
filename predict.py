"""
Run inference with a trained MoE NER model on custom text.

Usage:
    python predict.py --text "Barack Obama visited London last week."
    python predict.py --checkpoint checkpoints/moe_best.pt --text "Apple Inc. is based in Cupertino."
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import ID2LABEL, NUM_LABELS
from model import BiLSTMMoENER, BiLSTMNER


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    cfg = ckpt["cfg"]
    model_type = "moe" if "moe" in os.path.basename(ckpt_path) else "base"

    if model_type == "moe":
        model = BiLSTMMoENER(
            vocab_size=len(vocab),
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_labels=NUM_LABELS,
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
            expert_hidden_dim=cfg["expert_hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
        )
    else:
        model = BiLSTMNER(
            vocab_size=len(vocab),
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_labels=NUM_LABELS,
            num_layers=cfg["num_layers"],
            dropout=0.0,
        )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model, vocab


def predict(model, vocab, text: str, device: str):
    tokens = text.split()
    input_ids = torch.tensor(
        [[vocab.get(t, vocab["<UNK>"]) for t in tokens]], dtype=torch.long
    ).to(device)
    with torch.no_grad():
        logits = model(input_ids)
    pred_ids = logits.argmax(dim=-1)[0].tolist()
    return list(zip(tokens, [ID2LABEL[i] for i in pred_ids]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/moe_best.pt")
    parser.add_argument("--text", default="Barack Obama visited London last week .")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint, args.device)
    results = predict(model, vocab, args.text, args.device)

    print(f"\nInput : {args.text}")
    print(f"\n{'Token':<20} {'Label':<12}")
    print("-" * 32)
    for token, label in results:
        marker = "  ← ENTITY" if label != "O" else ""
        print(f"{token:<20} {label:<12}{marker}")


if __name__ == "__main__":
    main()
