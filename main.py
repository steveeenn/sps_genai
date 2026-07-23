from fastapi.responses import StreamingResponse
from io import BytesIO
from pathlib import Path

from helper_lib.generator import generate_sample_image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
# torch.set_num_threads(1)
import torchvision.transforms as transforms

# from app.bigram_model import BigramModel
from app.rnn_model import RNNTextGenerator
from app.embedding_model import EmbeddingModel
from app.energy_model import (
    EnergyModel,
    generate_energy_image,
    load_energy_checkpoint,
)
from app.diffusion_model import (
    create_diffusion_model,
    generate_diffusion_image,
    load_diffusion_checkpoint,
)
from helper_lib.model import get_model

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

rnn_model = RNNTextGenerator(corpus)
embedding_model = EmbeddingModel()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_dir = Path(__file__).resolve().parent

cnn_model = get_model("CNN")
checkpoint_path = project_dir / "checkpoints/model_epoch_010.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
except FileNotFoundError:
    pass

cnn_model.to(device)
cnn_model.eval()

gan_model = get_model("GAN")
gan_checkpoint_path = project_dir / "checkpoints/mnist_gan.pth"

try:
    gan_checkpoint = torch.load(gan_checkpoint_path, map_location=device)
    gan_model.load_state_dict(gan_checkpoint["model_state_dict"])
except FileNotFoundError:
    pass

gan_model.to(device)
gan_model.eval()

energy_model = EnergyModel().to(device)
energy_checkpoint_path = project_dir / "checkpoints/energy_model.pth"
energy_model_ready = load_energy_checkpoint(
    energy_model,
    energy_checkpoint_path,
    device,
)

diffusion_model = create_diffusion_model().to(device)
diffusion_checkpoint_path = (
    project_dir / "checkpoints/diffusion_model.pth"
)
diffusion_model_ready = load_diffusion_checkpoint(
    diffusion_model,
    diffusion_checkpoint_path,
    device,
)

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = rnn_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated_text = rnn_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(request.word)
    return {
        "word": request.word,
        "embedding": embedding,
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_index = predicted.item()

    return {
        "filename": file.filename,
        "predicted_class": class_names[predicted_index],
        "confidence": float(confidence.item()),
    }


@app.get("/generate_digit")
def generate_digit():
    image = generate_sample_image(gan_model, device)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/generate_energy")
def generate_energy(steps: int = 256, seed: int | None = None):
    if not energy_model_ready:
        raise HTTPException(
            status_code=503,
            detail="Energy model checkpoint is not available",
        )
    if steps < 1 or steps > 512:
        raise HTTPException(
            status_code=400,
            detail="steps must be between 1 and 512",
        )

    image = generate_energy_image(
        energy_model,
        device,
        steps=steps,
        seed=seed,
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/generate_diffusion")
def generate_diffusion(
    steps: int = 20,
    seed: int | None = None,
):
    if not diffusion_model_ready:
        raise HTTPException(
            status_code=503,
            detail="Diffusion model checkpoint is not available",
        )
    if steps < 1 or steps > 200:
        raise HTTPException(
            status_code=400,
            detail="steps must be between 1 and 200",
        )

    image = generate_diffusion_image(
        diffusion_model,
        device,
        diffusion_steps=steps,
        seed=seed,
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
