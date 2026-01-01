# app.py
import streamlit as st
import requests
from datetime import datetime

# ⚠️ 必須是第一個 Streamlit 指令！
st.set_page_config(
    page_title="🏇 賽馬入位預測",
    page_icon="🐎",
    layout="centered"
)

# ==============================
# 初始化 session_state
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []  # 儲存歷史紀錄

if "actual_weight" not in st.session_state:
    st.session_state.actual_weight = 56.0
if "declared_weight" not in st.session_state:
    st.session_state.declared_weight = 58.0
if "draw" not in st.session_state:
    st.session_state.draw = 3
if "win_odds" not in st.session_state:
    st.session_state.win_odds = 3.5
if "jockey_id" not in st.session_state:
    st.session_state.jockey_id = 1
if "trainer_id" not in st.session_state:
    st.session_state.trainer_id = 2

# ==============================
# 功能函式
# ==============================
def reset_inputs():
    """重置所有輸入為預設值"""
    st.session_state.actual_weight = 56.0
    st.session_state.declared_weight = 58.0
    st.session_state.draw = 3
    st.session_state.win_odds = 3.5
    st.session_state.jockey_id = 1
    st.session_state.trainer_id = 2

def add_to_history(input_data, result):
    """將預測結果加入歷史紀錄"""
    record = {
        "時間": datetime.now().strftime("%m-%d %H:%M"),
        "負磅": input_data["actual_weight"],
        "體重": input_data["declared_weight"],
        "檔位": input_data["draw"],
        "賠率": input_data["win_odds"],
        "騎師ID": input_data["jockey_id"],
        "練馬師ID": input_data["trainer_id"],
        "預測": result["prediction"],
        "機率": f"{result['top3_probability']:.1%}"
    }
    st.session_state.history.insert(0, record)  # 插入到最前面（最新在上）

# ==============================
# 頁面標題
# ==============================
st.title("🏇 賽馬入位預測系統")
st.markdown("輸入馬匹資料，AI 幫你預測是否能進入前三名！")

API_URL = "https://btr-horse-api.onrender.com/predict"

# ==============================
# 側邊欄：輸入表單
# ==============================
with st.sidebar:
    st.header("請輸入馬匹資料")
    
    # 所有輸入綁定到 session_state
    actual_weight = st.number_input(
        "實際負磅 (kg)", min_value=40.0, max_value=70.0,
        value=st.session_state.actual_weight, step=1.0,
        key="actual_weight"
    )
    declared_weight = st.number_input(
        "排位體重 (kg)", min_value=40.0, max_value=70.0,
        value=st.session_state.declared_weight, step=1.0,
        key="declared_weight"
    )
    draw = st.number_input(
        "檔位", min_value=1, max_value=14,
        value=st.session_state.draw,
        key="draw"
    )
    win_odds = st.number_input(
        "獨贏賠率", min_value=1.0, max_value=999.0,
        value=st.session_state.win_odds, step=0.1,
        key="win_odds"
    )
    jockey_id = st.number_input(
        "騎師 ID", min_value=0, max_value=200,
        value=st.session_state.jockey_id,
        key="jockey_id"
    )
    trainer_id = st.number_input(
        "練馬師 ID", min_value=0, max_value=200,
        value=st.session_state.trainer_id,
        key="trainer_id"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.button("🎯 預測", use_container_width=True)
    with col2:
        reset_button = st.button("🔄 重置", use_container_width=True, on_click=reset_inputs)

# ==============================
# 主內容區：處理預測
# ==============================
if predict_button:
    with st.spinner("AI 正在分析中..."):
        try:
            features = [
                float(actual_weight),
                float(declared_weight),
                int(draw),
                float(win_odds),
                int(jockey_id),
                int(trainer_id)
            ]
            
            response = requests.post(
                API_URL,
                json={"features": features},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                prob = result["top3_probability"]
                prediction = result["prediction"]
                
                # 顯示結果
                st.subheader("📊 預測結果")
                if prob >= 0.6:
                    st.success(f"✅ 預測：{prediction}（機率：{prob:.1%}）")
                elif prob >= 0.4:
                    st.warning(f"⚠️ 預測：{prediction}（機率：{prob:.1%}）")
                else:
                    st.error(f"❌ 預測：{prediction}（機率：{prob:.1%}）")
                
                # 加入歷史紀錄
                input_data = {
                    "actual_weight": actual_weight,
                    "declared_weight": declared_weight,
                    "draw": draw,
                    "win_odds": win_odds,
                    "jockey_id": jockey_id,
                    "trainer_id": trainer_id
                }
                add_to_history(input_data, result)
                
            else:
                st.error(f"❌ API 回應錯誤：{response.status_code}")
                
        except Exception as e:
            st.error(f"💥 發生錯誤：{str(e)}")

# ==============================
# 歷史紀錄區
# ==============================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 歷史紀錄")
    
    # 顯示清除按鈕
    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️ 清除紀錄"):
            st.session_state.history = []
            st.rerun()  # 重新整理頁面
    
    # 顯示表格（最新在上）
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

# ==============================
# 頁尾
# ==============================
st.markdown("---")
st.caption("本服務基於真實賽馬數據訓練的 AI 模型，僅供娛樂參考。理性投注，切勿沉迷。")