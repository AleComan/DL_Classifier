import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = [
    "efficientnet_b0", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
    "convnext_tiny", "convnext_small",
    "ensemble_convnext_small_efficientnet_b2",   # ← ensamble
]


# ──────────────────────────────────────────────────────────────
# Ensamble: promedia logits de convnext_small + efficientnet_b2
# ──────────────────────────────────────────────────────────────
class EnsembleModel(nn.Module):
    """
    Wrapper que contiene dos backbones y promedia sus logits en inferencia.
    Los clasificadores se exponen bajo los nombres 'classifier' para que
    freeze_backbone / unfreeze_all los detecten correctamente.
    """

    def __init__(self, num_classes: int, pretrained: bool, dropout: float):
        super().__init__()

        weights = "DEFAULT" if pretrained else None

        # ── ConvNeXt-Small ──────────────────────────────────
        self.convnext = models.convnext_small(weights=weights)
        in_features_cnx = self.convnext.classifier[2].in_features
        self.convnext.classifier[2] = nn.Linear(in_features_cnx, num_classes)

        # ── EfficientNet-B2 ─────────────────────────────────
        self.efficientnet = models.efficientnet_b2(weights=weights)
        in_features_eff = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features_eff, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.convnext(x) + self.efficientnet(x)) / 2.0


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int, pretrained: bool, dropout: float):
    weights = "DEFAULT" if pretrained else None

    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b2":
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b4":
        model = models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif backbone == "convnext_small":
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif backbone == "ensemble_convnext_small_efficientnet_b2":
        model = EnsembleModel(num_classes, pretrained, dropout)

    else:
        raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {SUPPORTED_BACKBONES}")

    return model


# ──────────────────────────────────────────────────────────────
# Freeze / unfreeze — sin cambios, funcionan sobre cualquier modelo
# ──────────────────────────────────────────────────────────────
def freeze_backbone(model: nn.Module, backbone: str):
    """Fase 1: congela todo excepto los clasificadores finales."""
    frozen = 0
    for name, param in model.named_parameters():
        is_head = any(k in name for k in ["classifier", "fc"])
        param.requires_grad = is_head
        if not is_head:
            frozen += 1
    print(f"  Parámetros congelados: {frozen}")


def unfreeze_all(model: nn.Module):
    """Fase 2: descongela toda la red."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {total:,}")