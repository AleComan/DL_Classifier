import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders
from model import build_model, freeze_backbone, unfreeze_all


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (normalizada)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def run_phase(phase, model, loaders, criterion, optimizer, scheduler,
              epochs, device, class_names, cfg, save_path):
    """Ejecuta un bloque de entrenamiento completo (fase 1 o 2)."""
    best_val_acc = 0.0
    best_preds, best_labels = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, preds, labels = eval_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        print(
            f"  [Fase {phase}] Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        wandb.log({
            "phase": phase,
            f"phase{phase}/train_loss": train_loss,
            f"phase{phase}/train_acc":  train_acc,
            f"phase{phase}/val_loss":   val_loss,
            f"phase{phase}/val_acc":    val_acc,
            f"phase{phase}/lr":         scheduler.get_last_lr()[0],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds, best_labels = preds, labels
            # En run_phase(), donde llamas a torch.save:
            torch.save({
                "epoch": epoch,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
                "backbone": wandb.config.get("model.backbone", cfg["model"]["backbone"]),
                "config": cfg,
            }, save_path)
            print(f"    ✓ Mejor modelo guardado (val_acc={val_acc:.3f})")

    return best_val_acc, best_preds, best_labels


def main():
    # ── Config base desde YAML ───────────────────────────────
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ── Init W&B (el sweep sobreescribe cfg si está activo) ──
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=cfg["wandb"]["run_name"],
        mode=cfg["wandb"].get("mode", "online"),
        config=cfg,
    )

    # Leer config final (puede haber sido sobreescrita por sweep)
    wcfg = wandb.config

    # Reconstruir cfg anidado desde wandb.config (que es plano con claves "a.b")
    def get(key, default=None):
        # intenta primero la clave con punto (formato sweep), luego anidada
        flat_key = key.replace(".", "_")  # wandb a veces aplana con _
        if key in wcfg:
            return wcfg[key]
        # navegar el dict anidado original
        parts = key.split(".")
        val = cfg
        for p in parts:
            val = val.get(p, default)
            if val is None:
                return default
        return val

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datos ────────────────────────────────────────────────
    loaders, class_names, class_weights = get_dataloaders(
        get("data.data_dir"),
        get("data.image_size"),
        get("data.batch_size"),
        get("data.num_workers"),
    )
    num_classes = len(class_names)
    print(f"Clases ({num_classes}): {class_names}\n")

    # ── Modelo ───────────────────────────────────────────────
    model = build_model(
        get("model.backbone"),
        num_classes,
        get("model.pretrained"),
        get("model.dropout"),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=get("train.label_smoothing"),
    )

    save_path = Path("../models/best_model.pth")
    save_path.parent.mkdir(exist_ok=True)

    wandb.watch(model, log_freq=100)

    # ── Fase 1 ───────────────────────────────────────────────
    print("── FASE 1: Entrenando solo el clasificador ──")
    freeze_backbone(model, get("model.backbone"))

    optimizer1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=get("train.lr_phase1"),
        weight_decay=get("train.weight_decay"),
    )
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=get("train.epochs_phase1"))

    best_acc_p1, preds_p1, labels_p1 = run_phase(
        phase=1, model=model, loaders=loaders,
        criterion=criterion, optimizer=optimizer1, scheduler=scheduler1,
        epochs=get("train.epochs_phase1"),
        device=device, class_names=class_names,
        cfg=cfg, save_path=save_path,
    )

    # ── Fase 2 ───────────────────────────────────────────────
    print("\n── FASE 2: Fine-tuning completo ──")
    unfreeze_all(model)

    optimizer2 = optim.AdamW(
        model.parameters(),
        lr=get("train.lr_phase2"),
        weight_decay=get("train.weight_decay"),
    )
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=get("train.epochs_phase2"))

    best_acc_p2, preds_p2, labels_p2 = run_phase(
        phase=2, model=model, loaders=loaders,
        criterion=criterion, optimizer=optimizer2, scheduler=scheduler2,
        epochs=get("train.epochs_phase2"),
        device=device, class_names=class_names,
        cfg=cfg, save_path=save_path,
    )

    # ── Resultados finales ───────────────────────────────────
    best_preds  = preds_p2  if best_acc_p2 >= best_acc_p1 else preds_p1
    best_labels = labels_p2 if best_acc_p2 >= best_acc_p1 else labels_p1

    print("\n=== Informe de clasificación (mejor modelo) ===")
    print(classification_report(best_labels, best_preds, target_names=class_names))

    cm_path = Path("../models/confusion_matrix.png")
    plot_confusion_matrix(best_labels, best_preds, class_names, cm_path)
    wandb.log({"confusion_matrix": wandb.Image(str(cm_path))})

    wandb.finish()
    print(f"\nModelo guardado en: {save_path}")

if __name__ == "__main__":
    main()