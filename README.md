# DL Classifier

A deep learning system for automatic classification of real estate images into 15 scene categories. The final model is a **soft-voting ensemble** of **EfficientNet-B2** and **ConvNeXt-Small**, achieving **~97.67% overall accuracy** on the validation set. Built with PyTorch, tracked with Weights & Biases, and deployed via a FastAPI + Streamlit stack.

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

The best checkpoint by validation accuracy is saved to `models/best_model.pth`. If it beats the current global best across all runs, it also updates the global best.

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

If the sweep configuration changes (e.g. new backbones added), create a new sweep — existing runs are preserved in W&B.

### Supported backbones

| Backbone | Params | Notes |
|----------|--------|-------|
| `efficientnet_b0` | 4.0M | Lightweight, strong baseline |
| `efficientnet_b2` | 7.7M | **Final ensemble member** — best accuracy/cost balance |
| `efficientnet_b3` | 10.7M | Good balance for this dataset size |
| `efficientnet_b4` | 17.6M | Heavier, may overfit with few images |
| `convnext_tiny` | 28.6M | Modern architecture, strong transfer learning |
| `convnext_small` | 50.2M | **Final ensemble member** — strongest single model |

---

## Ensemble

After completing the hyperparameter sweep, the two best-performing checkpoints were combined into a **soft-voting ensemble**:

| Model | Val accuracy (standalone) |
|-------|---------------------------|
| EfficientNet-B2 | ~97% |
| ConvNeXt-Small | ~98% |
| **Ensemble (avg logits)** | **~97.9%** |

The ensemble averages the raw softmax probabilities from both models before taking the argmax. Neither model needs to be retrained — the two `best_model.pth` checkpoints are loaded simultaneously at inference time.

**Why these two?**  
- EfficientNet-B2 and ConvNeXt-Small have different inductive biases (compound-scaled CNN vs. modern pure-convolution transformer-style), so their errors are partially uncorrelated.  
- Both are compact enough to run in parallel on a single consumer GPU (RTX 4070 Laptop) without OOM.

---

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
| `model.backbone` | `efficientnet_b0` | Backbone architecture (see supported backbones above) |
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

Or use the **Recargar modelo** button in the Streamlit sidebar.

### Streamlit sidebar

The sidebar shows at all times:

- Active backbone, val accuracy, run ID and epoch of the model currently loaded by the API
- List of available classes
- Reload button to hot-swap the model without restarting

### Classifying multiple images

The app supports uploading and classifying multiple images at once. Results are displayed in a 3-column grid with confidence score and a Top 5 expandable panel per image.

---

## Results

Results correspond to the **EfficientNet-B2 + ConvNeXt-Small ensemble** evaluated on the validation set.

### Global metrics

| Metric | Value |
|--------|-------|
| Overall accuracy (ensemble) | **97.67%** |
| Best individual backbone | `convnext_small` |
| Ensemble strategy | Soft voting (average softmax probabilities) |

### Per-class accuracy (from normalised confusion matrix)

Values on the diagonal of the normalised confusion matrix represent per-class recall (= accuracy for that class).

| Class | Accuracy |
|-------|---------|
| Bedroom | 96% |
| Coast | 98% |
| Forest | 98% |
| Highway | 99% |
| Industrial | 97% |
| Inside city | 95% |
| Kitchen | 97% |
| Living room | 99% |
| Mountain | 99% |
| Office | 100% |
| Open country | 95% |
| Store | 96% |
| Street | 100% |
| Suburb | 100% |
| Tall building | 98% |

### Confusion matrix

Normalised confusion matrix (rows = ground truth, columns = predicted). The model was evaluated on the validation split with the ensemble of EfficientNet-B2 and ConvNeXt-Small.

![Confusion matrix](confusion_matrix.png)

> The hardest class pairs are **Inside city ↔ Suburb** (2% confusion) and **Open country ↔ Coast** (3% confusion), which is expected given their visual similarity.

---

## W&B project

Experiments, sweep runs, and confusion matrices are tracked at:

`https://wandb.ai/your_username/DL-classifier`

---

## Requirements

- Python 3.10
- PyTorch + CUDA (tested on RTX 4070 Laptop, CUDA 11.8)
- See `environment.yml` for the full dependency list