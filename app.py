import re
import joblib
import pymorphy3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Corporate Document Routing Service",
    description="Сервис классификации и маршрутизации входящих корпоративных документов",
    version="1.0.0"
)

morph = pymorphy3.MorphAnalyzer()

try:
    pipeline = joblib.load("model_pipeline.joblib")
except Exception:
    pipeline = None

class DocumentRequest(BaseModel):
    text: str = Field(..., min_length=3, example="Прошу предоставить очередной оплачиваемый отпуск на 14 дней")

class DocumentResponse(BaseModel):
    status: str
    target_department: str
    confidence: float
    probabilities: dict[str, float]

def preprocess_text(text: str) -> str:
    clean = re.sub(r'[^а-яё\s]', ' ', str(text).lower())
    words = clean.split()
    lemmas = [morph.parse(w)[0].normal_form for w in words if len(w) > 2]
    return ' '.join(lemmas)

@app.get("/health", tags=["Мониторинг"])
def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена в память")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=DocumentResponse, tags=["Маршрутизация"])
def route_document(request: DocumentRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Сервис недоступен: модель не инициализирована")

    cleaned_input = preprocess_text(request.text)
    if not cleaned_input:
        raise HTTPException(status_code=400, detail="Текст не содержит информативных токенов")

    probabilities = pipeline.predict_proba([cleaned_input])[0]
    classes = pipeline.classes_

    best_idx = probabilities.argmax()
    best_dept = classes[best_idx]
    confidence = float(probabilities[best_idx])

    CONFIDENCE_THRESHOLD = 0.60
    routing_status = "AUTOMATIC" if confidence >= CONFIDENCE_THRESHOLD else "MANUAL_REVIEW_REQUIRED"
    target_dept = best_dept if confidence >= CONFIDENCE_THRESHOLD else "Operator_Verification"

    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, probabilities)}

    return DocumentResponse(
        status=routing_status,
        target_department=target_dept,
        confidence=round(confidence, 4),
        probabilities=prob_map
    )
