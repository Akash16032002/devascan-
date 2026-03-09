from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import io
import json

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# ------------------- FastAPI app -------------------

app = FastAPI()

@app.get("/")
def root():
    return {"message": "DevScan backend running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Gemini setup -------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. In PowerShell run:\n"
        "$env:GEMINI_API_KEY='YOUR_REAL_KEY_HERE'"
    )

client = genai.Client(api_key=GEMINI_API_KEY)


class TextRequest(BaseModel):
    text: str


# ------------------- Gemini Text Summary -------------------

def summarize_with_gemini(text: str) -> str:
    prompt = f"""
You are an assistant that reads Hindi/Devanagari text and writes a short summary in simple English.

Input text:
{text}

Requirements:
- Write 1–2 sentences.
- Use simple English.
- Do NOT translate word-for-word.
Return only the English summary.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=types.Content(
            role="user",
            parts=[types.Part.from_text(prompt)]
        )
    )

    return response.text.strip()


@app.post("/process")
def process_text(req: TextRequest):
    summary = summarize_with_gemini(req.text)

    return {
        "original_text": req.text,
        "summary_english": summary,
    }


# ------------------- OCR CNN MODEL -------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_ocr_model():
    model_path = os.path.join("models", "ocr_cnn.pth")
    labels_path = os.path.join("models", "ocr_labels.json")

    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    return model, class_names, transform


ocr_model = None
ocr_class_names = None
ocr_transform = None


# ------------------- Sloka Image Endpoint -------------------

from PIL import Image
import io

@app.post("/sloka-image")
async def sloka_from_image(file: UploadFile = File(...)):

    global ocr_model, ocr_class_names, ocr_transform

    if ocr_model is None:
        ocr_model, ocr_class_names, ocr_transform = load_ocr_model()

    image_bytes = await file.read()

    # 🔥 THIS IS THE KEY FIX
    pil_image = Image.open(io.BytesIO(image_bytes))

    prompt = """
If the image does NOT contain readable Devanagari text,
respond EXACTLY with:
NO_DEVANAGARI_TEXT

If Devanagari text is present, output STRICTLY:

DEVANAGARI_TEXT:
<only devanagari text>

ENGLISH_TRANSLATION:
<only English translation>
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            pil_image  # ✅ ONLY THIS WORKS
        ],
    )

    output = response.text.strip()

    if output == "NO_DEVANAGARI_TEXT":
        return {
            "recognized_text": "Couldn't detect any readable Devanagari text.",
            "translation": ""
        }

    dev_text = ""
    eng_text = ""

    if "DEVANAGARI_TEXT:" in output and "ENGLISH_TRANSLATION:" in output:
        parts = output.split("ENGLISH_TRANSLATION:")
        dev_text = parts[0].replace("DEVANAGARI_TEXT:", "").strip()
        eng_text = parts[1].strip()

    if not dev_text or len(dev_text) < 2:
        return {
            "recognized_text": "Couldn't detect any readable Devanagari text.",
            "translation": ""
        }

    return {
        "recognized_text": dev_text,
        "translation": eng_text
    }
