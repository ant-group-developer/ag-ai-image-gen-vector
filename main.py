import io
import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoImageProcessor, AutoModel

app = FastAPI(title="DINOv2 Image Embedding Server")

# Load model
print("Loading DINOv2 model...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"Model loaded on {device}")


def extract_embedding(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
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

        embedding = extract_embedding(image)

        return {
            "filename": file.filename,
            "dimensions": len(embedding),
            "embedding": embedding,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )