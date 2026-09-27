"""Dehqon AI modelini Hugging Face Spaces'da ishlatadigan FastAPI server.

Bu fayl ai/model_server.py bilan bir xil mantiqqa ega (_translate_class_name,
predict_image, _get_treatment) — faqat http.server o'rniga FastAPI ishlatadi,
shunda bulutda (internetda) doimiy ishlaydigan HTTP server sifatida joylashadi.

Kutilayotgan fayl tuzilmasi (shu papka ichida, app.py bilan bir qatorda):
  models/plant_disease.keras
  models/class_names.txt
  disease_data.json
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
KERAS_MODEL_PATH = MODEL_DIR / "plant_disease.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
DISEASE_DATA_PATH = ROOT / "disease_data.json"

IMAGE_SIZE = (224, 224)
MIN_RELIABLE_CONFIDENCE = 0.60
MIN_RELIABLE_MARGIN = 0.10

DISEASE_DATA = json.loads(DISEASE_DATA_PATH.read_text(encoding="utf-8"))
MODEL = tf.keras.models.load_model(KERAS_MODEL_PATH)
CLASS_NAMES = [line.strip() for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _clean_label(value: str) -> str:
    return value.replace("_", " ").replace("___", " ").replace("__", " ").replace("(", " ").replace(")", " ").replace("  ", " ").strip()


def _translate_class_name(class_name: str) -> tuple[str, str]:
    record = DISEASE_DATA.get(class_name, {})
    plant_raw, _, disease_raw = class_name.partition("___")
    plant = record.get("plant") or _clean_label(plant_raw)
    disease = record.get("disease") or _clean_label(disease_raw or plant_raw)
    if plant_raw.lower().startswith("apple"):
        plant = "Olma"
    elif plant_raw.lower().startswith("peach"):
        plant = "Shaftoli"
    elif plant_raw.lower().startswith("strawberry"):
        plant = "Qulupnay"
    elif plant_raw.lower().startswith("tomato"):
        plant = "Pomidor"
    elif plant_raw.lower().startswith("potato"):
        plant = "Kartoshka"
    elif plant_raw.lower().startswith("grape"):
        plant = "Uzum"
    elif "corn" in plant_raw.lower():
        plant = "Makkajo‘xori"
    elif "pepper" in plant_raw.lower():
        plant = "Bolgar qalampiri"
    elif "cherry" in plant_raw.lower():
        plant = "Gilos"
    elif "orange" in plant_raw.lower():
        plant = "Apelsin"
    elif plant_raw.lower().startswith("blueberry"):
        plant = "Ko‘k mevali buta"
    elif plant_raw.lower().startswith("raspberry"):
        plant = "Malina"
    elif plant_raw.lower().startswith("soybean"):
        plant = "Soya"
    elif plant_raw.lower().startswith("squash"):
        plant = "Qovoq"
    if disease_raw and not record.get("disease"):
        disease = disease_raw.replace("_", " ").replace("___", " ")
        disease = disease.replace("-", " ")
        disease = disease.replace("(Black Measles)", "qora mezl")
        disease = disease.replace("Leaf Mold", "barglarning mog‘orlanganligi")
        disease = disease.replace("Mosaic Virus", "mosaika virusi")
        disease = disease.replace("Yellow Leaf Curl Virus", "sariq barglarning burishishi virusi")
        disease = disease.replace("Bacterial Spot", "bakterial dog‘")
        disease = disease.replace("Bacterial spot", "bakterial dog‘")
        disease = disease.replace("Leaf Scorch", "barg kuyishi")
        disease = disease.replace("Leaf scorch", "barg kuyishi")
        disease = disease.replace("Cedar apple rust", "olma zang kasalligi")
        disease = disease.replace("Common rust", "oddiy zang")
        disease = disease.replace("Cercospora leaf spot Gray leaf spot", "serkospora barg dog‘lanishi")
        disease = disease.replace("Powdery mildew", "un shudringi")
        disease = disease.replace("Target Spot", "nishon dog‘")
        disease = disease.replace("Target spot", "nishon dog‘")
        disease = disease.replace("Spider mites Two-spotted spider mite", "o‘rgimchakkana")
        disease = disease.replace("Spider mites Two-spotted_spider_mite", "o‘rgimchakkana")
        disease = disease.replace("Spider mites Two-spotted", "o‘rgimchakkana")
        disease = disease.replace("Spider mites", "o‘rgimchakkana")
        disease = disease.replace("Tomato mosaic virus", "pomidor mozaika virusi")
        disease = disease.replace("Tomato Yellow Leaf Curl Virus", "pomidor sariq barg burishish virusi")
        disease = disease.replace("Septoria leaf spot", "septoriya barg dog‘lanishi")
        disease = disease.replace("Leaf Mold", "barg mog‘ori")
        disease = disease.replace("Leaf_Mold", "barg mog‘ori")
        disease = disease.replace("healthy", "sog‘lom")
        disease = disease.replace("Early blight", "erta chirish")
        disease = disease.replace("Late blight", "kech chirish")
        disease = disease.replace("scab", "qizarish")
        disease = disease.replace("rot", "chirish")
        disease = disease.replace("spot", "dog‘")
    return plant, disease


def _localized_prediction(class_name: str, confidence: float) -> dict[str, object]:
    plant, disease = _translate_class_name(class_name)
    return {
        "class_name": class_name,
        "plant": plant,
        "disease": disease,
        "label": f"{plant}: {disease}",
        "confidence": round(confidence * 100, 1),
    }


def _get_treatment(class_name: str) -> tuple[str, str]:
    record = DISEASE_DATA.get(class_name, DISEASE_DATA.get("defaults", {}))
    treatment = record.get("treatment") or DISEASE_DATA["defaults"]["treatment"]
    prevention = record.get("prevention") or DISEASE_DATA["defaults"]["prevention"]
    return treatment, prevention


def predict_image(image_bytes: bytes, plant_hint: str | None = None) -> dict[str, object]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMAGE_SIZE)
    pixels = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = MODEL.predict(pixels, verbose=0)[0]
    allowed_indices = [index for index, name in enumerate(CLASS_NAMES) if not plant_hint or name.lower().startswith(f"{plant_hint.lower()}___")]
    if not allowed_indices:
        allowed_indices = list(range(len(CLASS_NAMES)))
    candidate_probabilities = probabilities[allowed_indices]
    candidate_total = float(candidate_probabilities.sum())
    if candidate_total > 0:
        candidate_probabilities = candidate_probabilities / candidate_total
    order = np.argsort(candidate_probabilities)[::-1][:3]
    best_index = int(allowed_indices[int(order[0])])
    class_name = CLASS_NAMES[best_index]
    plant, disease = _translate_class_name(class_name)
    treatment, prevention = _get_treatment(class_name)
    confidence = float(candidate_probabilities[int(order[0])])
    margin = confidence - float(candidate_probabilities[int(order[1])])
    reliable = confidence >= MIN_RELIABLE_CONFIDENCE and margin >= MIN_RELIABLE_MARGIN
    if disease == "sog‘lom":
        treatment = "Kasallik belgisi topilmadi. O‘simlikni odatdagi parvarishda davom ettiring."
        prevention = "Me’yorida sug‘oring, barglarni kuzating va dalani toza saqlang."
    advice = (
        f"{plant} o‘simligida {disease} aniqlanadi. Davo: {treatment} Oldini olish: {prevention}"
        if reliable
        else "Rasm aniq ko‘rinmayapti. Bargni yaqinroqdan, yorug‘ joyda va toza holatda qayta suratga oling."
    )
    return {
        "class_name": class_name,
        "plant": plant if reliable else "Aniqlanmadi",
        "disease": disease if reliable else "Ishonchsiz natija",
        "confidence": round(confidence * 100, 1),
        "reliable": reliable,
        "margin": round(margin * 100, 1),
        "treatment": treatment,
        "prevention": prevention,
        "advice": advice,
        "top_predictions": [_localized_prediction(CLASS_NAMES[allowed_indices[int(index)]], float(candidate_probabilities[int(index)])) for index in order],
    }


app = FastAPI(title="Dehqon AI model server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    return {"status": "ok", "message": "Dehqon AI model server ishlayapti.", "class_count": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    try:
        image_bytes = await request.body()
        plant_hint = request.headers.get("x-plant-type") or None
        result = predict_image(image_bytes, plant_hint)
        return JSONResponse(result)
    except Exception as error:  # noqa: BLE001 - foydalanuvchiga tushunarli xato qaytarish uchun
        return JSONResponse({"error": f"Rasm tahlil qilinmadi: {error}"}, status_code=400)
