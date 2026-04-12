import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ["efficientnet_b0", "efficientnet_b2", "mobilenet_v3_small", "resnet50"]


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
    
    else:
        raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {SUPPORTED_BACKBONES}")

    return model


def freeze_backbone(model: nn.Module, backbone: str):
    """Fase 1: congela todo excepto el clasificador final."""
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