"""
CoNLL 2003 NER data: word ids (lower-cased, GloVe-aware), casing ids, character ids.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

PAD, UNK = 0, 1
NUM_CASE = 6  # pad, lower, UPPER, Title, digit-ish, other


def load_conll():
    return load_dataset("eriktks/conll2003", revision="refs/convert/parquet")


def case_id(tok: str) -> int:
    if any(c.isdigit() for c in tok):
        return 4
    if tok.islower():
        return 1
    if tok.isupper():
        return 2
    if tok[:1].isupper():
        return 3
    return 5


def build_vocabs(raw, glove_words: List[str]):
    """
    Word vocab = lower-cased train words seen >= 2 times, plus every GloVe word that occurs
    anywhere in train/val/test (using unlabeled pretrained vectors for dev/test words is the
    standard setup; no labels are involved). Char vocab comes from train only.
    """
    glove_set = set(glove_words)
    train_counts = Counter(t.lower() for ex in raw["train"] for t in ex["tokens"])
    seen_any = Counter()
    for split in ("train", "validation", "test"):
        for ex in raw[split]:
            seen_any.update(t.lower() for t in ex["tokens"])

    word2id = {"<PAD>": PAD, "<UNK>": UNK}
    for w, c in train_counts.items():
        if c >= 2:
            word2id.setdefault(w, len(word2id))
    for w in seen_any:
        if w in glove_set:
            word2id.setdefault(w, len(word2id))

    char_counts = Counter(c for ex in raw["train"] for t in ex["tokens"] for c in t)
    char2id = {"<PAD>": 0, "<UNK>": 1}
    for ch, n in char_counts.items():
        if n >= 2:
            char2id[ch] = len(char2id)
    return word2id, char2id


def build_embedding_matrix(word2id: Dict[str, int], glove_words: List[str], glove_vecs: np.ndarray, seed: int = 0):
    rng = np.random.RandomState(seed)
    dim = glove_vecs.shape[1]
    mat = rng.normal(0, 0.1, size=(len(word2id), dim)).astype("float32")
    mat[PAD] = 0.0
    g = {w: i for i, w in enumerate(glove_words)}
    hit = 0
    for w, i in word2id.items():
        if w in g:
            mat[i] = glove_vecs[g[w]]
            hit += 1
    return torch.tensor(mat), hit


class CoNLL2003Dataset(Dataset):
    def __init__(self, raw_split, word2id, char2id, max_len=128, max_word_len=20, train=False, word_dropout=0.0):
        self.examples = []
        for ex in raw_split:
            toks = ex["tokens"][:max_len]
            wids = [word2id.get(t.lower(), UNK) for t in toks]
            cids = [case_id(t) for t in toks]
            chars = [[char2id.get(c, 1) for c in t[:max_word_len]] for t in toks]
            self.examples.append((wids, cids, chars, ex["ner_tags"][:max_len]))
        self.train = train
        self.word_dropout = word_dropout
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def collate_fn(self, batch: List[Tuple]):
        B = len(batch)
        S = max(len(b[0]) for b in batch)
        W = max(len(c) for b in batch for c in b[2])
        words = torch.zeros(B, S, dtype=torch.long)
        cases = torch.zeros(B, S, dtype=torch.long)
        chars = torch.zeros(B, S, W, dtype=torch.long)
        labels = torch.full((B, S), -100, dtype=torch.long)
        mask = torch.zeros(B, S, dtype=torch.long)
        for i, (wids, cids, ch, labs) in enumerate(batch):
            n = len(wids)
            w = torch.tensor(wids)
            if self.train and self.word_dropout > 0:
                drop = torch.rand(n) < self.word_dropout
                w = torch.where(drop, torch.full_like(w, UNK), w)
            words[i, :n] = w
            cases[i, :n] = torch.tensor(cids)
            labels[i, :n] = torch.tensor(labs)
            mask[i, :n] = 1
            for j, c in enumerate(ch):
                chars[i, j, : len(c)] = torch.tensor(c)
        return words, cases, chars, labels, mask
