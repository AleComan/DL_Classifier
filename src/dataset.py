import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size: int, split: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),  # oculta parches aleatorios
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),  # estira directamente
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def compute_class_weights(dataset):
    """Calcula pesos inversamente proporcionales al tamaño de cada clase."""
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(dataset.classes)  # normalizar
    return weights


def get_dataloaders(data_dir: str, image_size: int, batch_size: int, num_workers: int):
    loaders = {}
    class_names = None
    class_weights = None

    for split in ["train", "val"]:
        path = Path(data_dir) / split
        if not path.exists():
            raise FileNotFoundError(f"No se encuentra: {path}")

        dataset = datasets.ImageFolder(
            path,
            transform=get_transforms(image_size, split)
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

        if class_names is None:
            class_names = dataset.classes
            class_weights = compute_class_weights(dataset)

    return loaders, class_names, class_weights