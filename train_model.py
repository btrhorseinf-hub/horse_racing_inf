import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

# ==============================
# 1. 載入所有 Excel 賽果檔案
# ==============================
def load_race_results(file_paths):
    all_races = []
    for file in file_paths:
        df = pd.read_excel(file)
        # 假設每場比賽以 "Race X" 開頭的行作為分隔（根據你提供的格式）
        # 此處簡化：假設 DataFrame 已經是「每匹馬一行」的結構
        # 若實際格式不同，需先清洗（見下方備註）
        all_races.append(df)
    return pd.concat(all_races, ignore_index=True)

# ==============================
# 2. 特徵工程
# ==============================
def engineer_features(df):
    # 複製避免修改原始數據
    df = df.copy()
    
    # 目標變量：是否跑入前三
    df['is_top3'] = df['名次'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    
    # 清理賠率（轉為數值）
    df['獨贏賠率'] = pd.to_numeric(df['獨贏賠率'], errors='coerce').fillna(999)  # 冷門設高值
    
    # 騎師 & 練馬師 → 轉為 ID（簡單編碼）
    df['jockey_id'] = pd.Categorical(df['騎師']).codes
    df['trainer_id'] = pd.Categorical(df['練馬師']).codes
    
    # 選擇特徵
    feature_cols = [
        '實際負磅', '排位體重', '檔位',
        '獨贏賠率', 'jockey_id', 'trainer_id'
    ]
    
    # 移除缺失值
    df = df.dropna(subset=feature_cols + ['is_top3'])
    
    return df, feature_cols

# ==============================
# 3. 訓練模型
# ==============================
def train_and_save_model(df, feature_cols, model_path="model/model.pkl"):
    X = df[feature_cols]
    y = df['is_top3']
    
    # 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 訓練隨機森林
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 評估
    y_pred = model.predict(X_test)
    print("✅ 模型準確率:", accuracy_score(y_test, y_pred))
    print("\n📊 分類報告:")
    print(classification_report(y_test, y_pred, target_names=["未入位", "入位"]))
    
    # 儲存模型
    Path(model_path).parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n💾 模型已儲存至: {model_path}")
    
    return model, feature_cols

# ==============================
# 主程式
# ==============================
if __name__ == "__main__":
    # 列出你的 Excel 檔案
    files = [
        "data/HKJ_local_results_20230910.xlsx",
        "data/HKJ_local_results_20230913.xlsx",
        "data/HKJ_local_results_20230917.xlsx",
        "data/HKJ_local_results_20230920.xlsx"
    ]
    
    print("📥 載入賽馬數據...")
    raw_df = load_race_results(files)
    
    print("🔧 特徵工程...")
    df, features = engineer_features(raw_df)
    
    print(f"📊 共 {len(df)} 筆有效樣本")
    print(f"🎯 特徵: {features}")
    
    print("\n🧠 訓練模型...")
    model, _ = train_and_save_model(df, features)
