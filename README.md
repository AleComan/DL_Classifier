# Real Estate Image Classifier

A deep learning system for automatic classification of real estate images into 15 scene categories. Built with PyTorch (EfficientNet transfer learning), tracked with Weights & Biases, and deployed via a FastAPI + Streamlit stack.

---

## Project structure

```
real-estate-classifier/
│
├── data/
│   ├── train/              # Training images organised by class folder
│   └── val/                # Validation images organised by class folder
│
├── src/
│   ├── config.yaml         # Central configuration (model, training, W&B)
│   ├── dataset.py          # DataLoader, transforms, class weight computation
│   ├── model.py            # Backbone selection and transfer learning setup
│   ├── train.py            # Two-phase training loop with W&B logging
│   └── sweep.yaml          # W&B hyperparameter sweep configuration
│
├── api/
│   └── main.py             # FastAPI inference service
│
├── app/
│   └── streamlit_app.py    # Streamlit front-end
│
├── models/
│   └── best_model.pth      # Saved checkpoint (created after training)
│
├── environment.yml
└── README.md
```

---

## Classes

The model classifies images into 15 categories:

`Bedroom` · `Coast` · `Forest` · `Highway` · `Industrial` · `Inside city` · `Kitchen` · `Living room` · `Mountain` · `Office` · `Open country` · `Store` · `Street` · `Suburb` · `Tall building`

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate DL
```

### 2. Configure W&B

```bash
wandb login
```

Paste your API key from [wandb.ai/authorize](https://wandb.ai/authorize). Then set your username in `src/config.yaml`:

```yaml
wandb:
  entity: "your_wandb_username"
```

### 3. Prepare the dataset

Organise images in `ImageFolder` format — one subfolder per class:

```
data/
├── train/
│   ├── Bedroom/
│   ├── Kitchen/
│   └── ...
└── val/
    ├── Bedroom/
    ├── Kitchen/
    └── ...
```

---

## Training

### Single run

```bash
conda activate DL
cd src
python train.py
```

Training runs in two phases:

- **Phase 1** — backbone frozen, only the classification head is trained (`epochs_phase1` epochs, `lr_phase1`)
- **Phase 2** — full network unfrozen, fine-tuned at a lower learning rate (`epochs_phase2` epochs, `lr_phase2`)

The best checkpoint by validation accuracy is saved automatically to `models/best_model.pth`.

### Hyperparameter sweep (W&B)

Create the sweep once:

```bash
cd src
wandb sweep --project real-estate-classifier sweep.yaml
```

Then launch the agent (run until satisfied, stop with `Ctrl+C`):

```bash
wandb agent your_username/real-estate-classifier/<sweep_id>
```

The sweep searches over backbone, dropout, learning rates, label smoothing, and batch size using Bayesian optimisation.

---

## Configuration reference

Edit `src/config.yaml` to change any training parameter:

| Key | Default | Description |
|-----|---------|-------------|
| `model.backbone` | `efficientnet_b0` | Backbone architecture (`efficientnet_b0`, `efficientnet_b2`, `mobilenet_v3_small`, `resnet50`) |
| `model.dropout` | `0.4` | Dropout rate before the classifier head |
| `model.pretrained` | `true` | Use ImageNet pretrained weights |
| `data.batch_size` | `32` | Batch size (reduce to `16` if GPU OOM) |
| `data.image_size` | `224` | Input resolution |
| `data.num_workers` | `0` | Always `0` on Windows |
| `train.epochs_phase1` | `10` | Epochs with backbone frozen |
| `train.epochs_phase2` | `20` | Epochs with full fine-tuning |
| `train.lr_phase1` | `0.001` | Learning rate for phase 1 |
| `train.lr_phase2` | `0.0001` | Learning rate for phase 2 |
| `train.label_smoothing` | `0.1` | Label smoothing factor |

---

## Deployment

Two terminals are required.

**Terminal 1 — FastAPI backend:**

```bash
conda activate DL
cd api
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Streamlit frontend:**

```bash
conda activate DL
cd app
streamlit run streamlit_app.py
```

Then open:

- Streamlit app → [http://localhost:8501](http://localhost:8501)
- API docs (Swagger) → [http://localhost:8000/docs](http://localhost:8000/docs)

### API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and model info |
| `GET` | `/classes` | List of available classes |
| `POST` | `/predict` | Classify an uploaded image |

**Example `/predict` response:**

```json
{
  "prediction": "Kitchen",
  "confidence": 0.9731,
  "top5": [
    { "class": "Kitchen",     "probability": 0.9731 },
    { "class": "Living room", "probability": 0.0142 },
    { "class": "Office",      "probability": 0.0087 },
    { "class": "Store",       "probability": 0.0031 },
    { "class": "Bedroom",     "probability": 0.0009 }
  ]
}
```

---

## Results

Best run (EfficientNet-B0, two-phase fine-tuning):

| Metric | Value |
|--------|-------|
| Validation accuracy | **96.1%** |
| Macro F1-score | **0.96** |
| Parameters | 4.0M |

Hardest classes: `Living room` (F1 0.91), `Inside city` (F1 0.93), `Open country` (F1 0.93) — visually similar to adjacent categories.

---

## W&B project

Experiments, sweep runs, and confusion matrices are tracked at:

`https://wandb.ai/your_username/real-estate-classifier`

Reviewers: `agascon@comillas.edu` and `rkramer@comillas.edu` have been invited to the workspace.

---

## Requirements

- Python 3.10
- PyTorch + CUDA (tested on RTX 4070 Laptop, CUDA 11.8)
- See `environment.yml` for the full dependency list
