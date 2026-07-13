import io
import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel

app = FastAPI(title="CLIP Image/Text Embedding Server")

# Load model
MODEL_NAME = "openai/clip-vit-large-patch14"

print(f"Loading CLIP model ({MODEL_NAME})...")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"Model loaded on {device}")


def _normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def extract_image_embedding(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    embedding = features.squeeze().cpu().numpy()
    embedding = _normalize(embedding)

    return embedding.tolist()


def extract_text_embedding(text: str) -> list[float]:
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    embedding = features.squeeze().cpu().numpy()
    embedding = _normalize(embedding)

    return embedding.tolist()


class TextEmbedRequest(BaseModel):
    text: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "model": MODEL_NAME,
    }


@app.post("/embed")
async def embed_image(
    file: UploadFile = File(...)
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        embedding = extract_image_embedding(image)

        return {
            "filename": file.filename,
            "dimensions": len(embedding),
            "embedding": embedding,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )


@app.post("/embed-text")
async def embed_text(
    body: TextEmbedRequest
):
    try:
        if not body.text or not body.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text must not be empty"
            )

        embedding = extract_text_embedding(body.text)

        return {
            "text": body.text,
            "dimensions": len(embedding),
            "embedding": embedding,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process text: {str(e)}"
        )