#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost model to predict top-3 finish (is_top3) in HKJC races.
Compatible with XGBoost >= 1.6
Author: Your Name
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report,
    confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(data_path: str):
    """讀取處理好的數據集"""
    logger.info(f"讀取數據: {data_path}")
    df = pd.read_parquet(data_path)
    
    # 驗證必要欄位
    required_cols = ['is_top3', 'actual_weight', 'declared_weight', 'draw', 'win_odds']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"缺失必要欄位: {missing}")
    
    logger.info(f"載入 {len(df)} 筆記錄，入位比例: {df['is_top3'].mean():.2%}")
    return df

def prepare_features(df: pd.DataFrame):
    """準備特徵與標籤"""
    # 選擇特徵（排除 ID 和元資料）
    feature_cols = [
        'actual_weight',
        'declared_weight',
        'draw',
        'win_odds',
        'horse_name_id',
        'jockey_id',
        'trainer_id'
    ]
    # 只保留存在的特徵
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df['is_top3'].astype(int)
    
    # 處理缺失值（用中位數填補數值特徵）
    numeric_cols = ['actual_weight', 'declared_weight', 'draw', 'win_odds']
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    # 分類 ID 缺失值設為 -1
    cat_cols = ['horse_name_id', 'jockey_id', 'trainer_id']
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna(-1).astype(int)
    
    logger.info(f"使用特徵: {list(X.columns)}")
    return X, y

def plot_feature_importance(model, feature_names, output_dir):
    """繪製特徵重要性"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()

def main(data_path: str, model_output_dir: str):
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 1. 讀取數據
    df = load_data(data_path)
    
    # 2. 準備特徵
    X, y = prepare_features(df)
    
    # 3. 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
    
    # 4. 訓練 XGBoost（兼容新版 XGBoost）
    logger.info("開始訓練 XGBoost 模型...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 注意：新版 XGBoost 不支援 fit() 中的 early_stopping_rounds
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # 5. 評估
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"測試集準確率: {acc:.4f}")
    logger.info(f"測試集 AUC: {auc:.4f}")
    logger.info("\n分類報告:\n" + classification_report(y_test, y_pred))
    
    # 6. 儲存模型
    model_path = os.path.join(model_output_dir, "xgb_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"模型已儲存至: {model_path}")
    
    # 7. 特徵重要性圖
    plot_feature_importance(model, list(X.columns), model_output_dir)
    logger.info(f"特徵重要性圖已儲存至: {model_output_dir}/feature_importance.png")
    
    # 8. 混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(model_output_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 XGBoost 賽馬入位預測模型")
    parser.add_argument(
        "--data", 
        default="data/processed/hkjc_racing_data.parquet",
        help="處理後的數據集路徑"
    )
    parser.add_argument(
        "--output", 
        default="models",
        help="模型輸出路徑"
    )
    
    args = parser.parse_args()
    main(args.data, args.output)