from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from .core import predict_from_bytes

app = FastAPI(
    title="Medical X-Ray Classification API",
    version="1.0.0",
    description="HTTP-сервис для классификации рентгеновских снимков",
    docs_url="/docs",
    redoc_url=None,
)


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Предсказанный класс")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели")
    probabilities: Dict[str, float] = Field(..., description="Вероятности по всем классам")
    latency_ms: float = Field(..., ge=0.0, description="Время инференса, мс")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "xray-classifier"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)):
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail="Ожидается изображение (image/jpeg, image/png)")


    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Файл пустой")
        result = predict_from_bytes(image_bytes)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")