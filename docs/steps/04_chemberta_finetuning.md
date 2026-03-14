# Step 4: ChemBERTa Fine-Tuning

## Purpose

Fine-tune a pretrained molecular language model from HuggingFace Hub for PC-SAFT parameter regression. Unlike Steps 1–2 where we engineered features from SMILES, here the model reads raw SMILES strings directly and leverages chemical representations learned from ~77 million molecules during pretraining.

This is the **transfer learning** play: a model that has "read" the chemical language should generalize better to novel chemistries (like HFOs) than one trained only on 1,800 molecules with hand-crafted features.

## What It Adds Over Step 3

| After Step 3 | After This Step |
|--------------|----------------|
| Features require RDKit descriptor computation | Model consumes raw SMILES strings directly |
| Representations learned from scratch on ~1,800 molecules | Pretrained on ~77M molecules, fine-tuned on 1,800 |
| No language model / transformer experience shown | Full HuggingFace workflow demonstrated |
| Feature engineering is a bottleneck for new molecule types | Tokenizer handles any valid SMILES automatically |

## Skills Demonstrated

- **HuggingFace `transformers` library**: `AutoModel`, `AutoTokenizer`, `Trainer`, `TrainingArguments`
- **Transfer learning**: Loading pretrained weights, freezing/unfreezing layers, learning rate warmup
- **Custom model heads**: Adding a regression head on top of a pretrained encoder
- **HuggingFace Hub**: Pushing a fine-tuned model as a public artifact
- **Tokenization**: Understanding SMILES tokenization vs. natural language tokenization

## Candidate Pretrained Models

| Model | Hub ID | Pretraining Data | Architecture |
|-------|--------|-----------------|--------------|
| ChemBERTa | `seyonec/ChemBERTa-zinc-base-v1` | 77M SMILES (ZINC) | RoBERTa-base |
| ChemBERTa-2 | `DeepChem/ChemBERTa-77M-MTR` | 77M SMILES | RoBERTa-base, multi-task |
| MoLFormer | `ibm/MoLFormer-XL-both-10pct` | 1.1B SMILES | Linear attention transformer |

Start with **ChemBERTa** (`seyonec/ChemBERTa-zinc-base-v1`) -- it's the simplest to fine-tune and has the most community examples.

## Implementation Guide

### 1. Project structure

```
model/
  hf/
    __init__.py
    chemberta_model.py    # Custom model with regression head
    train_chemberta.py    # Fine-tuning script
    dataset.py            # HuggingFace Dataset for PC-SAFT
```

### 2. Custom regression model (`model/hf/chemberta_model.py`)

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ChemBERTaForPCSAFT(nn.Module):
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for RoBERTa-base

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # m, sigma, epsilon_k
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regression_head(cls_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(predictions, labels)

        return {"loss": loss, "predictions": predictions}
```

### 3. Dataset preparation (`model/hf/dataset.py`)

```python
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PCSAFTSmilesDataset(Dataset):
    def __init__(self, smiles, targets, tokenizer, max_length=128):
        self.encodings = tokenizer(
            smiles, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(
            list(zip(targets["m"], targets["sigma"], targets["epsilon_k"])),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }
```

### 4. Fine-tuning script (`model/hf/train_chemberta.py`)

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="model/saved/chemberta",
    num_train_epochs=50,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,          # lower LR for fine-tuning pretrained weights
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
)
```

### 5. Custom metrics callback

```python
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    targets = ["m", "sigma", "epsilon_k"]
    metrics = {}
    for i, t in enumerate(targets):
        metrics[f"mae_{t}"] = mean_absolute_error(labels[:, i], predictions[:, i])
        metrics[f"r2_{t}"] = r2_score(labels[:, i], predictions[:, i])
    return metrics
```

### 6. Training strategies

**Option A: Full fine-tuning** (recommended for small models like ChemBERTa)

All parameters are trainable. Use a small learning rate (2e-5) with warmup.

**Option B: Freeze encoder, train head only**

```python
for param in model.encoder.parameters():
    param.requires_grad = False
```

Faster, but less expressive. Good as a first test to verify the pipeline works.

**Option C: Gradual unfreezing** (most sophisticated)

1. Freeze encoder, train head for 10 epochs
2. Unfreeze top 2 encoder layers, train for 20 more epochs
3. Unfreeze all layers, train for 20 more epochs with very small LR

This prevents catastrophic forgetting and is worth discussing in an interview even if you only implement Option A.

### 7. Register with the evaluation harness

Add a `ChemBERTaModel` to `model/registry.py` so `python -m model.evaluate --models rf nn chemberta` works out of the box.

### 8. Push to HuggingFace Hub (optional but impressive)

```python
model.encoder.push_to_hub("your-username/chemberta-pcsaft")
tokenizer.push_to_hub("your-username/chemberta-pcsaft")
```

Include a model card with dataset description, training hyperparameters, and evaluation metrics.

## Evaluation & Success Criteria

### Metrics to compare

| Model | R² (m) | R² (σ) | R² (ε/k) | MAE (m) | MAE (σ) | MAE (ε/k) |
|-------|--------|--------|----------|---------|---------|-----------|
| RF (combined) | — | — | — | — | — | — |
| NN (combined) | — | — | — | — | — | — |
| ChemBERTa | — | — | — | — | — | — |

### What success looks like

- ChemBERTa fine-tuning converges (loss decreases, no NaN gradients)
- The model produces reasonable predictions (within physical bounds: m > 0, σ > 0, ε/k > 0)
- Performance is competitive with or better than the NN, especially on ε/k
- Training completes in under 30 minutes on a single GPU (or ~2 hours on CPU)

### Realistic expectations

With only ~1,800 training samples, ChemBERTa may not dramatically outperform a well-tuned NN with Morgan fingerprints. The value here is demonstrating the workflow and the potential: if more data becomes available (ML-SAFT, experimental submissions), the pretrained model has the capacity to leverage it far more effectively.

### What to watch for

- **Overfitting**: ChemBERTa has ~85M parameters vs. ~1,800 samples. Monitor train vs. validation loss closely. Use weight decay, dropout, and early stopping aggressively.
- **Tokenization issues**: SMILES characters like `[`, `]`, `=`, `#` must be handled by the tokenizer. ChemBERTa's tokenizer was trained on SMILES, so this should work, but verify on a few examples.
- **Scale sensitivity**: The regression head outputs raw values; make sure target normalization is consistent with Steps 2–3.

### When to move to Step 5

You're ready for Step 5 when:

1. You have a three-way comparison table (RF vs. NN vs. ChemBERTa)
2. You can explain the trade-offs: "RF is fast and interpretable, NN captures more complex patterns, ChemBERTa leverages pretrained chemical knowledge"
3. You've chosen a "best model" to deploy (likely the NN or ChemBERTa) based on the metrics
4. The evaluation harness includes all three models seamlessly
