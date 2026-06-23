from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as transforms

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from model import get_model

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
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
cnn_model = get_model("CNN")
checkpoint_path = "checkpoints/model_epoch_010.pth"

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
except FileNotFoundError:
    pass

cnn_model.to(device)
cnn_model.eval()

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
    generated_text = bigram_model.generate_text(request.start_word, request.length)
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