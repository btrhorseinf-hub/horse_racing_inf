# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from io import StringIO
import plotly.express as px
import os

# ====== 設定頁面 ======
st.set_page_config(
    page_title="🏇 賽馬價值投注分析儀",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("🏇 賽馬價值投注分析儀")

# ====== 載入模型（使用快取避免重複載入）======
@st.cache_resource
def load_model_and_encoders():
    if not all(os.path.exists(f) for f in ["model.pkl", "jockey_encoder.pkl", "trainer_encoder.pkl"]):
        return None, None, None, "模型檔案不存在！請先訓練模型。"
    
    try:
        model = joblib.load("model.pkl")
        jockey_encoder = joblib.load("jockey_encoder.pkl")
        trainer_encoder = joblib.load("trainer_encoder.pkl")
        return model, jockey_encoder, trainer_encoder, None
    except Exception as e:
        return None, None, None, f"模型載入失敗：{str(e)}"

model, jockey_encoder, trainer_encoder, error_msg = load_model_and_encoders()

if error_msg:
    st.error(f"❌ {error_msg}")
    st.stop()

# ====== 初始化 SQLite 歷史資料庫 ======
HISTORY_DB = "history.db"

def init_history_db():
    conn = sqlite3.connect(HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_date TEXT,
            horse_name TEXT,
            jockey TEXT,
            trainer TEXT,
            win_odds REAL,
            predicted_top3_prob REAL,
            value_score REAL,
            kelly_fraction REAL
        )
    """)
    conn.commit()
    conn.close()

init_history_db()

def save_predictions_to_db(predictions):
    conn = sqlite3.connect(HISTORY_DB)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    for p in predictions:
        cursor.execute("""
            INSERT INTO predictions 
            (race_date, horse_name, jockey, trainer, win_odds, predicted_top3_prob, value_score, kelly_fraction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            p["horse_name"],
            p["jockey"],
            p["trainer"],
            p["win_odds"],
            p["predicted_top3_prob"],
            p["value_score"],
            p["kelly_fraction"]
        ))
    conn.commit()
    conn.close()

def get_all_predictions_from_db():
    conn = sqlite3.connect(HISTORY_DB)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df.to_dict(orient="records") if not df.empty else []

# ====== 計算函數 ======
def calculate_implied_probability(odds: float) -> float:
    return 1.0 / odds if odds > 0 else 0.0

def calculate_kelly_fraction(predicted_prob: float, odds: float) -> float:
    if odds <= 1 or predicted_prob <= 0:
        return 0.0
    estimated_win_prob = min(predicted_prob / 3.0, 0.99)
    b = odds - 1
    q = 1 - estimated_win_prob
    kelly = (b * estimated_win_prob - q) / b
    return max(0.0, min(kelly, 0.1))  # 上限 10%

# ====== 頁籤 ======
tab_predict, tab_history = st.tabs(["📊 預測", "📜 歷史紀錄"])

# ========== 頁籤 1：預測 ==========
with tab_predict:
    st.subheader("上傳賽事資料 CSV")
    st.info("📌 檔案需包含欄位：`horse_name`, `jockey`, `trainer`, `actual_weight`, `draw`, `win_odds`")
    
    uploaded_file = st.file_uploader("選擇 CSV 檔案", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            # 讀取 CSV
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            input_df = pd.read_csv(stringio)
            
            # 驗證必要欄位
            required_cols = ["horse_name", "jockey", "trainer", "actual_weight", "draw", "win_odds"]
            missing = [col for col in required_cols if col not in input_df.columns]
            if missing:
                st.error(f"❌ 缺少必要欄位：{missing}")
                st.stop()
            
            st.write("📄 上傳的資料：")
            st.dataframe(input_df, use_container_width=True)

            # 預處理：編碼
            df = input_df.copy()
            df['jockey_encoded'] = df['jockey'].apply(
                lambda x: jockey_encoder.transform([x])[0] if x in jockey_encoder.classes_ else -1
            )
            df['trainer_encoded'] = df['trainer'].apply(
                lambda x: trainer_encoder.transform([x])[0] if x in trainer_encoder.classes_ else -1
            )

            # 特徵矩陣
            feature_cols = ['actual_weight', 'draw', 'win_odds', 'jockey_encoded', 'trainer_encoded']
            X = df[feature_cols].fillna(-1)

            # 預測
            top3_probs = model.predict_proba(X)[:, 1]

            # 產生結果
            results = []
            for i, row in df.iterrows():
                win_odds = float(row['win_odds'])
                pred_prob = float(top3_probs[i])
                implied_prob = calculate_implied_probability(win_odds)
                value_score = pred_prob - implied_prob
                kelly_frac = calculate_kelly_fraction(pred_prob, win_odds)

                results.append({
                    "horse_name": row["horse_name"],
                    "jockey": row["jockey"],
                    "trainer": row["trainer"],
                    "win_odds": win_odds,
                    "predicted_top3_prob": pred_prob,
                    "implied_probability": implied_prob,
                    "value_score": value_score,
                    "kelly_fraction": kelly_frac
                })

            # 儲存到資料庫
            save_predictions_to_db(results)

            # 排序並顯示
            df_res = pd.DataFrame(results).sort_values(by="value_score", ascending=False).reset_index(drop=True)
            st.success("✅ 預測完成！以下是按「價值分數」排序的推薦名單：")

            # 顯示推薦卡片
            for _, r in df_res.iterrows():
                value = r["value_score"]
                kelly_pct = r["kelly_fraction"] * 100
                if value > 0.4:
                    st.markdown(f"""
                    <div style="border-left: 5px solid red; padding: 12px; margin: 12px 0; background-color: #fff5f5; border-radius: 4px;">
                        <h4>💥 {r['horse_name']}（{r['jockey']} / {r['trainer']}）</h4>
                        <p>💰 賠率：{r['win_odds']} | 價值分數：<b>{value:.3f}</b></p>
                        <p>🎯 模型預測 Top3 機率：{r['predicted_top3_prob']:.1%}</p>
                        <p><b>🔥 強烈建議下注！</b> 建議注碼：總資金的 <b>{kelly_pct:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif value > 0.2:
                    st.markdown(f"""
                    <div style="border-left: 5px solid green; padding: 12px; margin: 12px 0; background-color: #f0fff4; border-radius: 4px;">
                        <h4>✅ {r['horse_name']}（{r['jockey']} / {r['trainer']}）</h4>
                        <p>💰 賠率：{r['win_odds']} | 價值分數：<b>{value:.3f}</b></p>
                        <p>🎯 模型預測 Top3 機率：{r['predicted_top3_prob']:.1%}</p>
                        <p>值得考慮，建議注碼：總資金的 <b>{kelly_pct:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="padding: 12px; margin: 12px 0; border: 1px solid #eee; border-radius: 4px;">
                        <h4>⚪ {r['horse_name']}（{r['jockey']} / {r['trainer']}）</h4>
                        <p>💰 賠率：{r['win_odds']} | 價值分數：<b>{value:.3f}</b></p>
                        <p>模型預測 Top3 機率：{r['predicted_top3_prob']:.1%} → 無顯著價值</p>
                    </div>
                    """, unsafe_allow_html=True)

            # 圖表
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.bar(
                    df_res,
                    y='horse_name',
                    x=['predicted_top3_prob', 'implied_probability'],
                    title="模型 vs 市場機率對比",
                    barmode='group',
                    orientation='h',
                    labels={'value': '機率', 'variable': '來源'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.bar(
                    df_res,
                    y='horse_name',
                    x='value_score',
                    title="價值分數排行榜",
                    color='value_score',
                    orientation='h',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.exception(f"處理檔案時發生錯誤：{e}")

# ========== 頁籤 2：歷史紀錄 ==========
with tab_history:
    st.subheader("過去預測紀錄")
    try:
        history_data = get_all_predictions_from_db()
        if history_data:
            df_hist = pd.DataFrame(history_data)
            # 格式化數值
            df_hist["predicted_top3_prob"] = df_hist["predicted_top3_prob"].apply(lambda x: f"{x:.1%}")
            df_hist["value_score"] = df_hist["value_score"].round(4)
            df_hist["kelly_fraction"] = df_hist["kelly_fraction"].apply(lambda x: f"{x:.1%}")
            
            display_cols = [
                "race_date", "horse_name", "jockey", "trainer",
                "win_odds", "predicted_top3_prob", "value_score", "kelly_fraction"
            ]
            st.dataframe(df_hist[display_cols], use_container_width=True)
        else:
            st.info("尚無歷史紀錄。請先進行一次預測。")
    except Exception as e:
        st.error(f"讀取歷史紀錄失敗：{e}")