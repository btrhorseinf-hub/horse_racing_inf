from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import os

app = FastAPI(title="Horse Racing Prediction API")

# 載入模型（啟動時載入一次）
MODEL_PATH = "model/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ 模型不存在: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
FEATURE_NAMES = ['actual_weight', 'declared_weight', 'draw', 'win_odds', 'jockey_id', 'trainer_id']

class RaceInput(BaseModel):
    features: List[float]  # 順序必須與訓練一致！

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "horse-api"}

@app.post("/predict")
def predict(input_data: RaceInput):
    try:
        # 驗證輸入長度
        if len(input_data.features) != len(FEATURE_NAMES):
            raise HTTPException(
                status_code=400,
                detail=f"需要 {len(FEATURE_NAMES)} 個特徵，但收到 {len(input_data.features)}"
            )
        
        # 預測入位機率
        proba = model.predict_proba([input_data.features])[0]  # [未入位機率, 入位機率]
        is_top3_prob = float(proba[1])
        
        # 預測是否入位
        prediction = "入位" if is_top3_prob > 0.5 else "未入位"
        
        return {
            "prediction": prediction,
            "top3_probability": round(is_top3_prob, 4),
            "input_features": dict(zip(FEATURE_NAMES, input_data.features))
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")
