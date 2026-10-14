# Mixture-of-Experts layer for Named Entity Recognition

A Mixture-of-Experts (MoE) layer implemented from scratch in PyTorch, added to a BiLSTM sequence tagger and evaluated on CoNLL-2003 NER against a plain BiLSTM and a parameter-matched dense-FFN control.

## Model

```
word (GloVe 100d, lower-cased) + casing embedding (8d) + char-CNN (50 filters)
   -> BiLSTM x 2 (512 hidden, packed sequences)
   -> [optional block, residual + LayerNorm]
        none : plain BiLSTM baseline
        ffn  : dense MLP (256 hidden)               <- capacity control
        moe  : 4 expert MLPs (256 hidden), softmax gate, top-2 routing
   -> Linear -> 9 BIO labels
```

MoE details (`moe_layer.py`): learned linear gate -> softmax -> top-2 selection with renormalised weights; each expert is a 2-layer MLP with ReLU and dropout. An optional Switch-style load-balancing loss (`--lb_coef`) is computed over real tokens only (padding excluded), and per-expert routing shares are recorded for analysis.

## Results (CoNLL-2003 test, entity-level F1 via seqeval, 3 seeds, 8 epochs)

| Model | Params | Test F1 (mean ± std) | Test acc |
|---|---:|:---:|:---:|
| BiLSTM baseline | 4.78M | **89.42 ± 0.30** | 97.88% |
| + dense FFN (control) | 5.05M | 88.79 ± 0.55 | 97.75% |
| + MoE (4 experts, top-2) | 5.84M | 88.39 ± 0.53 | 97.65% |
| + MoE + load-balancing loss (0.01) | 5.84M | 89.33 ± 0.39 | 97.87% |

Per-seed scores are in `results/`. Model selection uses validation F1 only; test F1 is computed once on the best-validation checkpoint.

**Findings**
- MoE with a load-balancing loss matches the baseline (89.33 vs 89.42; the gap is well inside seed-to-seed noise). It does not beat it.
- Without the balance loss MoE is about 1 F1 worse than the baseline; with it, about 0.9 F1 better. Expert shares then stay in the 17-33% range, so routing does not collapse.
- Adding a dense FFN of similar size does not help either, so extra capacity alone is not the limiting factor for this tagger.
- 8 experts with the balance loss (seed 0 only) scored 89.73 against 89.76 for 4 experts.

## Reproduce

```bash
pip install -r requirements.txt
python prepare_glove.py            # one-time: GloVe 100d -> glove_vecs.npy, glove_words.txt
python train.py --models none ffn moe --seeds 0 1 2 --epochs 8
python train.py --models moe --seeds 0 1 2 --epochs 8 --lb_coef 0.01
```

Useful flags: `--num_experts`, `--lb_coef`, `--hidden_dim`, `--dropout`, `--device cpu|mps|cuda`. Results are appended to a `.jsonl` file (`--out`), including per-expert routing shares for MoE runs.

## Files

- `moe_layer.py` - Expert and MoELayer (gating, top-k routing, balance loss, usage stats)
- `model.py` - NERTagger (embeddings, char-CNN, BiLSTM, optional FFN/MoE block)
- `dataset.py` - CoNLL-2003 loading, vocab, casing and char features
- `train.py` - training, whole-split seqeval F1, multi-seed runs
- `predict.py` - inference on custom text (uses the original pipeline; see note below)

> `predict.py` has not yet been updated to the GloVe/char-CNN model above.
