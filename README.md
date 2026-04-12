# DL Classifier

A deep learning system for automatic classification of real estate images into 15 scene categories. Built with PyTorch (EfficientNet transfer learning), tracked with Weights & Biases, and deployed via a FastAPI + Streamlit stack.

---

## Project structure

```
DL-classifier/
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
│   ├── best_model.pth      # Global best checkpoint (updated automatically)
│   └── model_<run_id>.pth  # Per-run checkpoints (one per sweep run)
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

Images can have variable sizes and aspect ratios — the pipeline handles resizing automatically to 224x224.

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

The best checkpoint by validation accuracy is saved to `models/best_model.pth`. If it is the best run seen so far across all runs, it also updates the global best.

### Hyperparameter sweep (W&B Bayesian optimisation)

Create the sweep once:

```bash
cd src
wandb sweep --project DL-classifier sweep.yaml
```

Then launch the agent (stop anytime with Ctrl+C):

```bash
wandb agent your_username/DL-classifier/<sweep_id>
```

The sweep searches over backbone, dropout, learning rates, label smoothing, and batch size using Bayesian optimisation. Each run saves its own checkpoint to `models/model_<run_id>.pth` and updates `models/best_model.pth` only if it beats the current global best.

### Recovering a checkpoint from a previous run

If a better run was overwritten, checkpoints can be recovered in two ways:

**From local W&B files:**
```bash
dir "...\DL_Classifier\src\wandb" /s /b | findstr "best_model.pth"
```
Then copy the relevant file:
```bash
copy "...\wandb\run-<id>\files\best_model.pth" "...\models\best_model.pth"
```

**From W&B cloud:** go to the run in [wandb.ai](https://wandb.ai) -> Files tab -> download `best_model.pth`.

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
| `wandb.mode` | `online` | Set to `offline` to disable cloud sync |

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

- Streamlit app -> http://localhost:8501
- API docs (Swagger) -> http://localhost:8000/docs

### API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and model info |
| `GET` | `/classes` | List of available classes |
| `POST` | `/predict` | Classify an uploaded image |
| `POST` | `/reload` | Hot-reload the model from disk without restarting |

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

### Reloading the model without restarting

After replacing `models/best_model.pth` with a better checkpoint, reload the API without downtime:

```bash
curl -X POST http://localhost:8000/reload
```

Or use the **Reload model** button in the Streamlit sidebar.

### Identifying the active model

The Streamlit sidebar shows the run ID, backbone, validation accuracy, and epoch of the checkpoint currently loaded by the API — useful when running multiple sweep experiments.

---

## Results

> Results below reflect the best run found so far. Update after sweeps complete.

| Metric | Value |
|--------|-------|
| Validation accuracy | **XX.X%** |
| Macro F1-score | **X.XX** |
| Best backbone | `efficientnet_bX` |
| Best dropout | `X.X` |
| Best lr phase 1 | `X.XXXX` |
| Best lr phase 2 | `X.XXXXX` |
| Sweep runs completed | **XX** |

**Per-class F1-score (best run):**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Bedroom | X.XX | X.XX | X.XX |
| Coast | X.XX | X.XX | X.XX |
| Forest | X.XX | X.XX | X.XX |
| Highway | X.XX | X.XX | X.XX |
| Industrial | X.XX | X.XX | X.XX |
| Inside city | X.XX | X.XX | X.XX |
| Kitchen | X.XX | X.XX | X.XX |
| Living room | X.XX | X.XX | X.XX |
| Mountain | X.XX | X.XX | X.XX |
| Office | X.XX | X.XX | X.XX |
| Open country | X.XX | X.XX | X.XX |
| Store | X.XX | X.XX | X.XX |
| Street | X.XX | X.XX | X.XX |
| Suburb | X.XX | X.XX | X.XX |
| Tall building | X.XX | X.XX | X.XX |

---

## W&B project

Experiments, sweep runs, and confusion matrices are tracked at:

`https://wandb.ai/your_username/DL-classifier`

Reviewers: `agascon@comillas.edu` and `rkramer@comillas.edu` have been invited to the workspace.

---

## Requirements

- Python 3.10
- PyTorch + CUDA (tested on RTX 4070 Laptop, CUDA 11.8)
- See `environment.yml` for the full dependency list
