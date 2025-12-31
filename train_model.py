# train_model.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def main():
    # 設定路徑
    data_dir = "data"
    model_dir = "model"
    model_path = os.path.join(model_dir, "model.pkl")
    
    # 檢查 data 資料夾是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 錯誤: '{data_dir}' 資料夾不存在！")
        print("請先建立 'data' 資料夾，並將你的 Excel 檔案放入其中。")
        return
    
    # 列出所有 .xlsx 檔案
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    if not excel_files:
        print(f"❌ 錯誤: '{data_dir}' 中沒有 .xlsx 檔案！")
        return
    
    print(f"📥 找到 {len(excel_files)} 個 Excel 檔案: {excel_files}")
    
    # 合併所有數據
    all_data = []
    for file in excel_files:
        try:
            df = pd.read_excel(os.path.join(data_dir, file))
            # 清理欄位名稱（移除前後空格）
            df.columns = df.columns.str.strip()
            all_data.append(df)
            print(f"✅ 已載入: {file} ({len(df)} 筆記錄)")
        except Exception as e:
            print(f"⚠️ 跳過 {file}: {e}")
    
    if not all_data:
        print("❌ 沒有成功載入任何數據！")
        return
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 總共合併 {len(df)} 筆賽馬記錄")
    
    # 必要欄位（根據你提供的檔案）
    required_cols = ["名次", "實際 負磅", "排位 體重", "檔位", "獨贏 賠率", "騎師", "練馬師"]
    
    # 檢查欄位是否存在
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必要欄位: {missing_cols}")
        print("可用欄位:", list(df.columns))
        return
    
    # 只保留必要欄位
    df = df[required_cols].copy()
    
    # 清理數據
    df = df.dropna(subset=["名次"])  # 移除名次缺失
    df["名次"] = pd.to_numeric(df["名次"], errors="coerce")
    df = df.dropna(subset=["名次"])
    
    # 目標變量：是否入前三
    df["is_top3"] = df["名次"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    
    # 處理賠率
    df["獨贏 賠率"] = pd.to_numeric(df["獨贏 賠率"], errors="coerce")
    df["獨贏 賠率"] = df["獨贏 賠率"].fillna(999)  # 冷門馬設高值
    
    # 騎師 & 練馬師編碼
    df["jockey_id"] = pd.Categorical(df["騎師"]).codes
    df["trainer_id"] = pd.Categorical(df["練馬師"]).codes
    
    # 特徵欄位
    feature_cols = ["實際 負磅", "排位 體重", "檔位", "獨贏 賠率", "jockey_id", "trainer_id"]
    df = df.dropna(subset=feature_cols)
    
    print(f"🔧 有效訓練樣本數: {len(df)}")
    
    if len(df) < 10:
        print("❌ 數據太少，無法訓練模型！")
        return
    
    # 準備訓練數據
    X = df[feature_cols]
    y = df["is_top3"]
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 訓練模型
    print("🧠 訓練隨機森林模型...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 評估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 測試準確率: {acc:.2%}")
    print("\n📊 分類報告:")
    print(classification_report(y_test, y_pred, target_names=["未入位", "入位"]))
    
    # 儲存模型
    Path(model_dir).mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n💾 模型已儲存至: {model_path}")
    print("\n🎉 訓練完成！現在可以部署 API 了。")

if __name__ == "__main__":
    main()