"""
Tag custom text with a trained NER model (checkpoints are written by train.py).

Usage:
    python predict.py --checkpoint checkpoints/moe_seed0.pt --text "Barack Obama visited London last week ."
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset import ID2LABEL, NUM_LABELS, UNK, case_id
from model import NERTagger


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    n_words, n_chars = len(ckpt["word2id"]), len(ckpt["char2id"])
    model = NERTagger(
        torch.zeros(n_words, ckpt["emb_dim"]),      # real vectors come from the state dict
        n_chars,
        NUM_LABELS,
        head=ckpt["head"],
        hidden_dim=ckpt["hidden_dim"],
        num_experts=ckpt["num_experts"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    return model.eval().to(device), ckpt["word2id"], ckpt["char2id"]


def predict(model, word2id, char2id, text: str, device: str, max_word_len: int = 20):
    tokens = text.split()
    words = torch.tensor([[word2id.get(t.lower(), UNK) for t in tokens]])
    cases = torch.tensor([[case_id(t) for t in tokens]])
    width = min(max(len(t) for t in tokens), max_word_len)
    chars = torch.zeros(1, len(tokens), width, dtype=torch.long)
    for i, t in enumerate(tokens):
        ids = [char2id.get(c, 1) for c in t[:max_word_len]]
        chars[0, i, : len(ids)] = torch.tensor(ids)
    mask = torch.ones(1, len(tokens), dtype=torch.long)
    with torch.no_grad():
        logits = model(words.to(device), cases.to(device), chars.to(device), mask.to(device))
    return list(zip(tokens, [ID2LABEL[i] for i in logits.argmax(-1)[0].tolist()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/moe_seed0.pt")
    parser.add_argument("--text", default="Barack Obama visited London last week .")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, word2id, char2id = load_model(args.checkpoint, args.device)
    results = predict(model, word2id, char2id, args.text, args.device)

    print(f"\nInput : {args.text}")
    print(f"\n{'Token':<20} {'Label':<12}")
    print("-" * 32)
    for token, label in results:
        marker = "  <- ENTITY" if label != "O" else ""
        print(f"{token:<20} {label:<12}{marker}")


if __name__ == "__main__":
    main()
