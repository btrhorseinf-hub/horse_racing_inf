#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
賽馬「是否進入前三名」預測模型訓練腳本 (v2)
輸入：data/historical_races_with_features.csv
輸出：
  - models/race_model_v2.pkl
  - data/predictions_train.csv
  - plots/feature_importance_v2.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ===== 設定路徑 =====
INPUT_DATA = "data/historical_races_with_features.csv"
MODEL_SAVE_PATH = "models/race_model_v2.pkl"
PREDICTIONS_OUTPUT = "data/predictions_train.csv"
FEATURE_IMPORTANCE_PLOT = "plots/feature_importance_v2.png"

# ===== 特徵選擇 =====
FEATURES = [
    'draw',                      # 檔位
    'actual_weight',            # 實際負磅
    'win_odds',                 # 獨贏賠率
    'race_distance',            # 距離
    'last_is_top3',             # 上場是否前三
    'top3_rate_last_1',         # 近1場前三率
    'top3_rate_last_3',         # 近3場前三率
    'top3_rate_last_5',         # 近5場前三率
    'avg_odds_last_3',          # 近3場平均賠率
    'avg_actual_weight_last_3', # 近3場平均負磅
    'days_since_last_race'      # 距離上場天數
]

TARGET = 'is_top3'

def prepare_data(df):
    """準備訓練資料：選取特徵、處理缺失值"""
    # 選取特徵與目標
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    
    # 處理缺失值：數值型用中位數填補
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_imputed, y, imputer

def train_model(X_train, y_train, model_type='rf'):
    """訓練指定模型"""
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'  # 處理不平衡類別
        )
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
    elif model_type == 'lr':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        return model, scaler
        # 注意：LogisticRegression 需要單獨處理 scaler
    else:
        raise ValueError("model_type must be 'rf', 'xgb', or 'lr'")
    
    model.fit(X_train, y_train)
    return model, None

def evaluate_model(model, X_test, y_test, model_type='rf', scaler=None):
    """評估模型並繪製特徵重要性"""
    # 預測
    if model_type == 'lr' and scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 評估指標
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✅ 模型評估結果 (AUC): {auc:.4f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))
    
    # 特徵重要性
    if model_type in ['rf', 'xgb']:
        importances = model.feature_importances_
    elif model_type == 'lr':
        importances = np.abs(model.coef_[0])
    
    feature_imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importance ({model_type.upper()})')
    plt.tight_layout()
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi=150)
    plt.close()
    print(f"📊 特徵重要性圖已儲存至: {FEATURE_IMPORTANCE_PLOT}")
    
    return y_pred_proba, feature_imp

def main():
    # 建立目錄
    Path("models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    
    # 讀取資料
    print(f"讀取訓練資料: {INPUT_DATA}")
    if not Path(INPUT_DATA).exists():
        print("❌ 請先執行 feature_engineering.py 生成特徵檔案！")
        return
    
    df = pd.read_csv(INPUT_DATA)
    print(f"總筆數: {len(df)}, 正樣本比例: {df[TARGET].mean():.2%}")
    
    # 準備資料
    X, y, imputer = prepare_data(df)
    
    # 切分訓練/測試集（按日期？這裡先隨機切分）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 訓練模型（可改為 'xgb' 或 'lr'）
    MODEL_TYPE = 'rf'  # ← 可在此切換模型
    print(f"\n🚀 開始訓練 {MODEL_TYPE.upper()} 模型...")
    
    if MODEL_TYPE == 'lr':
        model, scaler = train_model(X_train, y_train, MODEL_TYPE)
        joblib.dump((model, scaler, imputer, FEATURES), MODEL_SAVE_PATH)
    else:
        model, _ = train_model(X_train, y_train, MODEL_TYPE)
        joblib.dump((model, None, imputer, FEATURES), MODEL_SAVE_PATH)
    
    print(f"✅ 模型已儲存至: {MODEL_SAVE_PATH}")
    
    # 評估
    y_pred_proba, feature_imp = evaluate_model(
        model, X_test, y_test, MODEL_TYPE,
        scaler if MODEL_TYPE == 'lr' else None
    )
    
    # 儲存訓練集預測結果（供價值投注分析）
    if MODEL_TYPE == 'lr':
        X_full_scaled = scaler.transform(X)
        full_pred_proba = model.predict_proba(X_full_scaled)[:, 1]
    else:
        full_pred_proba = model.predict_proba(X)[:, 1]
    
    df_pred = df.copy()
    df_pred['predicted_top3_prob'] = full_pred_proba
    df_pred.to_csv(PREDICTIONS_OUTPUT, index=False, encoding='utf-8')
    print(f"💾 完整預測結果已儲存至: {PREDICTIONS_OUTPUT}")
    
    # 顯示 top 5 重要特徵
    print("\n🏆 Top 5 重要特徵:")
    print(feature_imp.head().to_string(index=False))

if __name__ == "__main__":
    main()