# api.py

import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from history import save_predictions  # 新增：歷史紀錄模組

# 初始化 FastAPI 應用
app = FastAPI(title="賽馬預測 API", description="預測馬匹進入前三名的機率與投注價值")

# 啟用 CORS（允許 Streamlit 前端跨域請求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入模型與編碼器
try:
    model = joblib.load("model.pkl")
    jockey_encoder = joblib.load("jockey_encoder.pkl")
    trainer_encoder = joblib.load("trainer_encoder.pkl")
    print("✅ 模型與編碼器載入成功！")
except Exception as e:
    print(f"⚠️ 模型載入失敗：{e}")
    model = None
    jockey_encoder = None
    trainer_encoder = None


def calculate_implied_probability(odds: float) -> float:
    """由賠率計算隱含勝率"""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def calculate_kelly_fraction(predicted_prob: float, odds: float) -> float:
    """
    優化版凱利公式（假設獨贏機率 ≈ Top3 機率 / 3）
    """
    if odds <= 1 or predicted_prob <= 0:
        return 0.0

    # 粗略估計獨贏機率（可調整）
    estimated_win_prob = min(predicted_prob / 3.0, 0.99)
    b = odds - 1
    q = 1 - estimated_win_prob
    kelly = (b * estimated_win_prob - q) / b
    return max(0.0, min(kelly, 0.1))  # 上限 10% 防過度投注


@app.get("/")
async def root():
    return {"message": "賽馬預測 API 正常運行中"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="未提供檔案")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="檔案內容為空")
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="檔案編碼錯誤，請使用 UTF-8 無 BOM 格式")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV 檔案為空")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法解析 CSV：{str(e)}")

    required_columns = ["horse_name", "jockey", "trainer", "actual_weight", "draw", "win_odds"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"缺少必要欄位：{missing_cols}")

    df = df.copy()
    df['jockey_encoded'] = df['jockey'].apply(
        lambda x: jockey_encoder.transform([x])[0] if jockey_encoder and x in jockey_encoder.classes_ else -1
    )
    df['trainer_encoded'] = df['trainer'].apply(
        lambda x: trainer_encoder.transform([x])[0] if trainer_encoder and x in trainer_encoder.classes_ else -1
    )

    feature_columns = ['actual_weight', 'draw', 'win_odds', 'jockey_encoded', 'trainer_encoded']
    X = df[feature_columns].fillna(-1)

    if model is None:
        raise HTTPException(status_code=500, detail="模型未載入，無法進行預測")

    try:
        top3_probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型預測失敗：{str(e)}")

    results = []
    for i, row in df.iterrows():
        win_odds = float(row['win_odds'])
        predicted_prob = float(top3_probs[i])
        implied_prob = calculate_implied_probability(win_odds)
        value_score = predicted_prob - implied_prob
        kelly_frac = calculate_kelly_fraction(predicted_prob, win_odds)

        results.append({
            "horse_name": row["horse_name"],
            "jockey": row["jockey"],
            "trainer": row["trainer"],
            "win_odds": win_odds,
            "predicted_top3_prob": round(predicted_prob, 4),
            "implied_probability": round(implied_prob, 4),
            "value_score": round(value_score, 4),
            "kelly_fraction": round(kelly_frac, 4)
        })

    # ✅ 自動儲存到歷史紀錄
    save_predictions(results)

    return {"predictions": results}
