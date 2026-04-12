import io
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from model import build_model

# ── Configuración ────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Real Estate Image Classifier",
    description="Clasifica imágenes inmobiliarias en 15 categorías usando EfficientNet.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carga del modelo al arrancar ─────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el modelo en {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    cfg = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    # Detectar backbone real por número de keys en state_dict
    num_keys = len(state_dict.keys())
    if num_keys > 480:
        backbone = "efficientnet_b2"
    elif num_keys > 300:
        backbone = "efficientnet_b0"
    else:
        backbone = "mobilenet_v3_small"

    print(f"Backbone detectado: {backbone} ({num_keys} keys)")

    model = build_model(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=False,
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model, class_names, cfg["data"]["image_size"]

model, CLASS_NAMES, IMAGE_SIZE = load_model()

# ── Transform de inferencia ──────────────────────────────────
def get_inference_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

transform = get_inference_transform(IMAGE_SIZE)


# ── Endpoints ────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "model": "EfficientNet-B0",
        "classes": len(CLASS_NAMES),
        "device": str(DEVICE),
    }


@app.get("/classes", summary="Lista de clases disponibles")
def get_classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict", summary="Clasifica una imagen")
async def predict(file: UploadFile = File(...)):
    # Validar que es imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"El archivo debe ser una imagen. Recibido: {file.content_type}"
        )

    # Leer y procesar imagen
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inferencia
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

    top5_probs, top5_idx = torch.topk(probs, k=5)

    return {
        "prediction": CLASS_NAMES[probs.argmax().item()],
        "confidence": round(probs.max().item(), 4),
        "top5": [
            {
                "class": CLASS_NAMES[idx.item()],
                "probability": round(prob.item(), 4),
            }
            for prob, idx in zip(top5_probs, top5_idx)
        ],
    }