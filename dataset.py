"""
Dataset utilities for CoNLL 2003 NER via HuggingFace datasets.
Handles tokenisation, label alignment, and batching.
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import Counter


# CoNLL 2003 tag set (BIO)
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


def build_vocab(dataset, min_freq: int = 1) -> Dict[str, int]:
    """Build word-level vocabulary from training split."""
    counter: Counter = Counter()
    for example in dataset["train"]:
        counter.update(example["tokens"])
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


class CoNLL2003Dataset(Dataset):
    def __init__(
        self,
        split: str,
        vocab: Dict[str, int],
        max_len: int = 128,
    ):
        raw = load_dataset("conll2003", trust_remote_code=True)[split]
        self.examples = []
        for ex in raw:
            tokens = ex["tokens"]
            ner_tags = ex["ner_tags"]
            input_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens][:max_len]
            labels = ner_tags[:max_len]
            self.examples.append((input_ids, labels))
        self.max_len = max_len
        self.pad_id = vocab["<PAD>"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return input_ids, labels

    def collate_fn(self, batch: List[Tuple[List[int], List[int]]]):
        input_ids_list, labels_list = zip(*batch)
        max_len = max(len(x) for x in input_ids_list)

        padded_ids, padded_labels, masks = [], [], []
        for ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)   # -100 ignored by CrossEntropy
            masks.append([1] * len(ids) + [0] * pad_len)

        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )
