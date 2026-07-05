# Mixture of Experts Layer for Named Entity Recognition

A custom **Mixture of Experts (MoE)** layer integrated into a **BiLSTM** sequence tagger, evaluated on **CoNLL 2003 NER**.

## Architecture

```
Token IDs
   │
   ▼
Embedding (128-d)
   │
   ▼
BiLSTM × 2 layers  (256-d hidden, bidirectional)
   │
   ▼
MoE Layer ──────────────────────────────────────┐
│  Gate: Linear → Softmax → Top-K selection     │
│  Experts: 4 × MLP(256 → 256 → 256)           │
│  Output: weighted sum of Top-2 expert outputs │
└────────────────────────────────────────────────┘
   │ (residual + LayerNorm)
   ▼
Linear → num_labels (9)
```

### MoE Layer Details

- **N = 4** expert MLPs, each a 2-layer network with ReLU and Dropout
- **Top-K = 2** routing: gate scores computed via a learned linear layer + Softmax, top-2 experts selected and re-normalised
- Residual connection wraps the MoE block: `out = LayerNorm(lstm_out + moe_out)`

## Results on CoNLL 2003 (test set)

| Model         | Test Accuracy | Test F1  |
|---------------|:-------------:|:--------:|
| BiLSTM (base) |    0.9406     |  0.7190  |
| BiLSTM + MoE  |    0.9374     |  0.7289  |
| **Δ (MoE)**   |  **−0.32 pts** | **+0.99 pts** |

> Single run, no fixed seed — a mixed result, not a clean win: MoE narrowly improves entity-F1 but not accuracy. Run `python train.py` to reproduce; numbers will vary run to run. An earlier version of this README reported a larger improvement (+1.2%/+3.2%) — that number predates a bug fix to `moe_layer.py` (the MoE forward pass crashed before the fix and was never actually reproducible) and should be disregarded.

## Setup

```bash
# 1. Clone
git clone https://github.com/Levi0p/-moe-ner.git
cd -moe-ner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train both models and compare
python train.py

# 4. Train only MoE
python train.py --model moe

# 5. Train only baseline
python train.py --model base
```

### Optional flags

```
--epochs       Number of training epochs  (default: 15)
--batch_size   Batch size                 (default: 32)
--lr           Learning rate              (default: 1e-3)
--num_experts  Number of MoE experts      (default: 4)
--top_k        Top-k routing              (default: 2)
```

## Inference

```bash
python predict.py \
  --checkpoint checkpoints/moe_best.pt \
  --text "Sundar Pichai visited New York to meet Microsoft executives."
```

Output:
```
Input : Sundar Pichai visited New York to meet Microsoft executives.

Token                Label
--------------------------------
Sundar               B-PER       ← ENTITY
Pichai               I-PER       ← ENTITY
visited              O
New                  B-LOC       ← ENTITY
York                 I-LOC       ← ENTITY
to                   O
meet                 O
Microsoft            B-ORG       ← ENTITY
executives.          O
```

## Project Structure

```
-moe-ner/
├── moe_layer.py   # MoE layer (Expert + MoELayer classes)
├── model.py       # BiLSTMNER + BiLSTMMoENER
├── dataset.py     # CoNLL 2003 loader + vocab builder
├── train.py       # Training loop + evaluation + comparison
├── predict.py     # Inference on custom text
├── checkpoints/   # Saved model weights (auto-created)
├── requirements.txt
└── README.md
```

## Dataset

[CoNLL 2003](https://huggingface.co/datasets/conll2003) is loaded automatically via HuggingFace `datasets`. No manual download required.

Labels: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`

## Key Design Choices

- **Word-level tokenisation** (no sub-word) keeps the model simple and interpretable.
- **Residual + LayerNorm** around MoE prevents representation collapse.
- **Top-K soft routing** (not hard routing) keeps the gating differentiable throughout training.
- **seqeval** used for entity-level F1 (span-based, not token-based) — the correct metric for NER.
