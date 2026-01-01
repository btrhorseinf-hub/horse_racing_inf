import streamlit as st
import requests
import json

# ======================
# 設定頁面基本資訊
# ======================
st.set_page_config(
    page_title="賽馬入位預測系統 🐎",
    page_icon="🏇",
    layout="wide"
)

# ======================
# 樣式美化
# ======================
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 應用標題與說明
# ======================
st.title("🏇 賽馬入位預測系統")
st.markdown("輸入馬匹資料，AI 幫你預測是否能進入前三名！")

# ======================
# 初始化 session_state
# ======================
if 'history' not in st.session_state:
    st.session_state.history = []

# ======================
# 輸入表單（左側）
# ======================
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📊 輸入馬匹資料")
    
    actual_weight = st.number_input("實際負磅 (kg)", min_value=0.0, value=55.0, step=0.5)
    declared_weight = st.number_input("排位體重 (kg)", min_value=0.0, value=500.0, step=1.0)
    draw = st.number_input("檔位", min_value=1, max_value=14, value=5)
    win_odds = st.number_input("獨贏賠率", min_value=1.0, value=5.0, step=0.1)
    jockey_id = st.number_input("騎師 ID", min_value=1, value=1)
    trainer_id = st.number_input("練馬師 ID", min_value=1, value=1)

    # 預測按鈕
    predict_button = st.button("🎯 預測", use_container_width=True)

# ======================
# 處理預測邏輯（帶 Loading Spinner）
# ======================
API_URL = "https://btr-horse-api.onrender.com/predict"  # ← 替換為你的 Render URL

if predict_button:
    # 驗證輸入
    if win_odds <= 0:
        st.error("❌ 獨贏賠率必須大於 0")
    else:
        # 準備發送資料
        data = {
            "actual_weight": actual_weight,
            "declared_weight": declared_weight,
            "draw": draw,
            "win_odds": win_odds,
            "jockey_id": jockey_id,
            "trainer_id": trainer_id
        }

        try:
            # 👇 關鍵：加入 Loading Spinner
            with st.spinner('🧠 正在連接 AI 模型，請稍候...'):
                response = requests.post(API_URL, json=data, timeout=10)
                
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                probability = result["probability"]

                # 顯示結果
                with col2:
                    st.subheader("✅ 預測結果")
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-box">🟢 <b>入位</b>（機率：{probability:.1%}）</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-box">🔴 <b>未能入位</b>（機率：{1 - probability:.1%}）</div>', unsafe_allow_html=True)

                # 儲存到歷史紀錄
                record = {
                    "input": data,
                    "output": {"prediction": prediction, "probability": probability}
                }
                st.session_state.history.insert(0, record)  # 插入最前面

            else:
                with col2:
                    st.error(f"⚠️ API 回應錯誤：{response.status_code}")
                    st.code(response.text)

        except requests.exceptions.Timeout:
            with col2:
                st.error("⏰ 請求超時！請檢查網路或稍後再試。")
        except requests.exceptions.ConnectionError:
            with col2:
                st.error("🔌 無法連接到 AI 伺服器！請確認 API URL 是否正確。")
        except Exception as e:
            with col2:
                st.error(f"💥 發生未知錯誤：{str(e)}")

# ======================
# 顯示歷史紀錄（右側）
# ======================
with col2:
    st.subheader("📜 歷史紀錄")
    
    if st.session_state.history:
        # 清除按鈕
        if st.button("🗑️ 清除紀錄", key="clear"):
            st.session_state.history = []
            st.rerun()

        # 顯示最近 5 筆
        for i, rec in enumerate(st.session_state.history[:5]):
            inp = rec["input"]
            out = rec["output"]
            pred_text = "🟢 入位" if out["prediction"] == 1 else "🔴 未入位"
            prob = f"{out['probability']:.1%}"
            
            st.markdown(f"""
            <div style="background:#f9f9f9; padding:10px; border-radius:8px; margin-bottom:10px;">
                <b>#{i+1}</b> 檔{inp['draw']} | 賠率{inp['win_odds']} | {pred_text} ({prob})
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("尚無預測紀錄。點擊左側「🎯 預測」開始使用！")

# ======================
# 頁尾
# ======================
st.markdown("---")
st.caption("© 2026 賽馬入位預測系統 | Powered by Streamlit + FastAPI")
