# train_xgb_model.py —— 完整版：含 Optuna 調參 + 儲存 feature_names.pkl

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import optuna

# 全域變數（供 objective 函數使用）
X_global = None
y_global = None

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, val_idx in cv.split(X_global, y_global):
        X_train_fold, X_val_fold = X_global.iloc[train_idx], X_global.iloc[val_idx]
        y_train_fold, y_val_fold = y_global.iloc[train_idx], y_global.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=0,
        )
        y_pred = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def main():
    global X_global, y_global

    print("🔄 正在讀取 historical_races.csv...")
    df = pd.read_csv("historical_races.csv")
    print(f"📊 原始資料形狀: {df.shape}")

    # 移除非必要欄位
    cols_to_drop = ["race_date", "horse_name"]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"✅ 已移除欄位: {cols_to_drop}")

    # 分離特徵與目標變數（關鍵！目標是 'is_top3'）
    if "is_top3" not in df.columns:
        raise ValueError("❌ 資料中缺少 'is_top3' 欄位！")

    y = df["is_top3"]          # 👈 正確的目標變數
    X = df.drop(columns=["is_top3"])  # 👈 移除目標變數

    # 編碼類別變數
    categorical_cols = ["jockey", "trainer", "track_condition", "class"]
    encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = X[col].fillna("未知").astype(str)
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # 確保無 object 型態
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        raise ValueError(f"❌ 仍有 object 型欄位: {object_cols}")

    print(f"✅ 最終特徵矩陣形狀: {X.shape}")
    print("使用特徵:", list(X.columns))

    # 設定全域變數供 Optuna 使用
    X_global = X
    y_global = y

    # ====== 超參數調優 ======
    print("\n🔍 開始 Optuna 超參數調優（目標：最大化 AUC）...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # 可調整試驗次數

    print(f"\n🎯 最佳 AUC: {study.best_value:.4f}")
    print("最佳參數:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ====== 訓練最終模型 ======
    best_params = study.best_params
    best_params.update({
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    })

    print("\n🚀 使用最佳參數訓練最終模型...")
    final_model = XGBClassifier(**best_params)
    final_model.fit(X, y)

    # 評估全資料 AUC
    y_pred_full = final_model.predict_proba(X)[:, 1]
    full_auc = roc_auc_score(y, y_pred_full)
    print(f"✅ 最終模型 AUC (全資料): {full_auc:.4f}")

    # ====== 儲存所有必要檔案 ======
    joblib.dump(final_model, "model.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")  # 👈 關鍵！供 SHAP 和 Streamlit 使用

    print("\n💾 已儲存:")
    print("   - model.pkl")
    print("   - label_encoders.pkl")
    print("   - feature_names.pkl")

if __name__ == "__main__":
    main()
