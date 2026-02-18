#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict top-3 probability for horses in a new race using trained XGBoost model.
Input: CSV file with horse info for a single race.
Output: Sorted list of horses with predicted probabilities.
Author: Your Name
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== 欄位映射（與 build_dataset.py 一致）=====
INPUT_COLUMN_MAPPING = {
    '馬名': 'horse_name',
    '騎師': 'jockey',
    '練馬師': 'trainer',
    '實際負磅': 'actual_weight',
    '排位體重': 'declared_weight',
    '檔位': 'draw',
    '獨贏賠率': 'win_odds',
    # 支援英文欄位（可選）
    'Horse': 'horse_name',
    'Jockey': 'jockey',
    'Trainer': 'trainer',
    'Weight': 'actual_weight',
    'Draw': 'draw',
    'Odds': 'win_odds',
}

def load_model_and_mappings(model_path: str):
    """載入模型與編碼映射（從訓練數據推斷）"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"✅ 載入模型: {model_path}")
    
    # 從模型獲取特徵順序
    expected_features = model.feature_names_in_
    logger.info(f"模型預期特徵: {list(expected_features)}")
    
    return model, expected_features

def prepare_input_data(input_df: pd.DataFrame, expected_features):
    """將輸入數據轉換為模型所需格式"""
    df = input_df.copy()
    
    # 1. 應用欄位映射
    df = df.rename(columns=INPUT_COLUMN_MAPPING)
    
    # 2. 驗證必要欄位
    required_cols = ['horse_name', 'jockey', 'trainer', 'actual_weight', 'win_odds']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"輸入數據缺少必要欄位: {missing}. 請確認 CSV 包含: {required_cols}")
    
    # 3. 數值轉換
    numeric_cols = ['actual_weight', 'declared_weight', 'draw', 'win_odds']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. 處理缺失數值（用中位數或默認值）
    if 'declared_weight' in df.columns and df['declared_weight'].isna().any():
        df['declared_weight'] = df['declared_weight'].fillna(df['declared_weight'].median())
    if 'draw' in df.columns and df['draw'].isna().any():
        df['draw'] = df['draw'].fillna(df['draw'].median())
    
    # 5. 生成 ID 編碼（模擬訓練時的 Categorical 編碼）
    # 注意：未知類別設為 -1（與訓練一致）
    cat_mappings = {}
    for cat_col in ['horse_name', 'jockey', 'trainer']:
        if f"{cat_col}_id" in expected_features:
            # 假設我們無法取得完整映射 → 用臨時編碼（僅限本次預測）
            df[f"{cat_col}_id"] = pd.Categorical(df[cat_col]).codes
    
    # 6. 選取模型需要的特徵
    available_features = [f for f in expected_features if f in df.columns]
    X = df[available_features].copy()
    
    # 7. 確保特徵順序與訓練時一致
    X = X.reindex(columns=expected_features, fill_value=-1)  # 未知特徵填 -1
    
    logger.info(f"準備好 {len(X)} 匹馬的預測數據")
    return X, df[['horse_name', 'jockey', 'trainer', 'win_odds']]

def main(input_csv: str, model_path: str, output_csv: str = None):
    # 1. 載入模型
    model, expected_features = load_model_and_mappings(model_path)
    
    # 2. 讀取輸入數據
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"輸入文件不存在: {input_csv}")
    
    input_df = pd.read_csv(input_csv)
    logger.info(f"讀取輸入數據: {input_csv} ({len(input_df)} 匹馬)")
    
    # 3. 準備特徵
    X, meta_df = prepare_input_data(input_df, expected_features)
    
    # 4. 預測
    proba = model.predict_proba(X)[:, 1]  # 入位機率 (class=1)
    predictions = pd.DataFrame({
        'horse_name': meta_df['horse_name'],
        'jockey': meta_df['jockey'],
        'trainer': meta_df['trainer'],
        'win_odds': meta_df['win_odds'],
        'top3_probability': proba
    })
    
    # 5. 排序（高機率在前）
    predictions = predictions.sort_values('top3_probability', ascending=False).reset_index(drop=True)
    predictions['rank'] = predictions.index + 1
    
    # 6. 輸出
    if output_csv:
        predictions.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"預測結果已儲存至: {output_csv}")
    
    # 7. 列印到終端（美觀格式）
    print("\n" + "="*70)
    print("🏇 賽馬入位機率預測結果")
    print("="*70)
    for _, row in predictions.iterrows():
        print(f"#{row['rank']:2d} | 機率: {row['top3_probability']:.2%} | "
              f"馬: {row['horse_name']} | 騎師: {row['jockey']} | "
              f"賠率: {row['win_odds']:.1f}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="預測新賽事馬匹入位機率")
    parser.add_argument(
        "--input",
        required=True,
        help="輸入 CSV 路徑（包含一場比賽的所有馬匹資訊）"
    )
    parser.add_argument(
        "--model",
        default="models/xgb_model.pkl",
        help="訓練好的模型路徑"
    )
    parser.add_argument(
        "--output",
        help="輸出預測結果 CSV 路徑（可選）"
    )
    
    args = parser.parse_args()
    main(args.input, args.model, args.output)