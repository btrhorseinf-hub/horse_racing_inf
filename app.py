# app.py
import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os

# ======================
# 📁 頁面設定
# ======================
st.set_page_config(
    page_title="🏇 賽馬入前三預測系統",
    page_icon="🐎",
    layout="wide"
)

st.title("🏇 賽馬入前三預測系統")
st.markdown("""
基於香港賽馬歷史數據訓練的機器學習模型，預測馬匹進入前三名的機率。
""")

# ======================
# 🔒 載入模型與編碼器
# ======================
@st.cache_resource
def load_model_and_encoders():
    paths = {
        'model': "models/model.pkl",
        'encoders': "models/label_encoders.pkl",
        'features': "models/feature_names.pkl"
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            st.error(f"❌ {name} 檔案不存在: {path}")
            st.stop()
    
    with open(paths['model'], 'rb') as f:
        model = pickle.load(f)
    with open(paths['encoders'], 'rb') as f:
        label_encoders = pickle.load(f)
    with open(paths['features'], 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, label_encoders, feature_names

model, label_encoders, feature_names = load_model_and_encoders()

# ======================
# 🎛️ 使用者輸入
# ======================
st.sidebar.header("🏇 請輸入參數")

jockey_options = sorted(label_encoders['jockey'].classes_)
trainer_options = sorted(label_encoders['trainer'].classes_)

selected_jockey = st.sidebar.selectbox("騎師", jockey_options)
selected_trainer = st.sidebar.selectbox("練馬師", trainer_options)
weight = st.sidebar.slider("實際負重 (kg)", 100, 140, 122)
barrier = st.sidebar.number_input("檔位", min_value=1, max_value=14, value=5, step=1)
win_odds = st.sidebar.number_input("🔹 獨贏賠率 (Win Odds)", min_value=1.0, max_value=200.0, value=5.0, step=0.1)
race_distance = st.sidebar.selectbox("🏁 賽程距離 (米)", [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], index=3)

# ======================
# 🧪 預處理輸入
# ======================
def preprocess_input(jockey, trainer, weight, barrier, win_odds, race_distance):
    try:
        jockey_encoded = label_encoders['jockey'].transform([jockey])[0]
        trainer_encoded = label_encoders['trainer'].transform([trainer])[0]
    except ValueError as e:
        st.warning(f"⚠️ 騎師或練馬師不在訓練資料中: {e}")
        st.stop()
    
    input_df = pd.DataFrame({
        'jockey': [jockey_encoded],
        'trainer': [trainer_encoded],
        'actual_weight': [weight],
        'draw': [barrier],
        'win_odds': [win_odds],
        'race_distance': [race_distance]
    })
    return input_df[feature_names]

input_df = preprocess_input(selected_jockey, selected_trainer, weight, barrier, win_odds, race_distance)

# ======================
# 🔮 預測與結果
# ======================
if st.sidebar.button("🚀 預測"):
    proba = model.predict_proba(input_df)[0]
    top3_prob = proba[1] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 預測結果")
        st.metric("入前三機率", f"{top3_prob:.2f}%")
        
        if top3_prob >= 70:
            st.success("✅ 高機會入前三！")
        elif top3_prob >= 40:
            st.warning("🔶 中等機會")
        else:
            st.error("🔴 低機會入前三")
    
    with col2:
        st.subheader("🔍 關鍵影響因素 (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # === 智能判斷格式 ===
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_vals = shap_values[1][0]
            else:
                shap_vals = shap_values[0]
            
            if len(shap_vals) != len(feature_names):
                raise ValueError("SHAP 值長度不符")
            
            # === 文字版 SHAP（最穩健）===
            name_map = {
                'jockey': '騎師',
                'trainer': '練馬師',
                'actual_weight': '實際負重',
                'draw': '檔位',
                'win_odds': '獨贏賠率',
                'race_distance': '賽程距離'
            }
            
            indices = np.argsort(np.abs(shap_vals))[::-1][:5]
            for i in indices:
                feat = feature_names[i]
                val = shap_vals[i]
                display_name = name_map.get(feat, feat)
                color = "green" if val > 0 else "red"
                sign = "+" if val > 0 else "-"
                st.markdown(f"• **{display_name}**: <span style='color:{color}'>{sign}{abs(val):.2f}</span>", unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"⚠️ SHAP 分析失敗: {str(e)[:100]}")

# ======================
# 📈 特徵重要性
# ======================
st.markdown("---")
st.subheader("📈 特徵重要性（模型訓練階段）")
try:
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    name_map = {
        'jockey': '騎師',
        'trainer': '練馬師',
        'actual_weight': '實際負重',
        'draw': '檔位',
        'win_odds': '獨贏賠率',
        'race_distance': '賽程距離'
    }
    importance_df['feature'] = importance_df['feature'].map(name_map).fillna(importance_df['feature'])
    st.bar_chart(importance_df.set_index('feature'))
except Exception as e:
    st.write("無法載入特徵重要性:", e)

# ======================
# ℹ️ 使用說明
# ======================
st.markdown("""
---
💡 **使用說明**：
- **數據來源**：香港賽馬會 (HKJC) 歷史賽果
- **目標定義**：`is_top3 = 1` 表示名次 ≤ 3
- **賠率查詢**：開跑前最後賠率可於 HKJC App 查看

🔧 **技術細節**：XGBoost 分類模型 + SHAP 可解釋性 AI  
✅ 相容 Hugging Face Spaces（無 IPython 或索引錯誤）
""")