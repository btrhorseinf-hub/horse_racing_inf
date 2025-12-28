from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Horse Racing Prediction API", version="1.0")

class RaceInput(BaseModel):
    features: List[float]  # 例如：[馬匹年齡, 過去勝率, 騎師評分, ...]

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "horse-api"}

@app.post("/predict")
def predict(input_data: RaceInput):
    # 這裡未來會載入模型做預測
    # 目前先返回 mock 結果
    return {
        "prediction": "horse_3",
        "probabilities": [0.1, 0.2, 0.5, 0.1, 0.1],
        "input_features": input_data.features
    }
