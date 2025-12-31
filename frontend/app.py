import streamlit as st
import requests
import time

# ==============================
# 配置
# ==============================
API_URL = "https://btr-horse-api.onrender.com/predict"
PAGE_TITLE = "🏇 馬匹勝率預測系統"
PAGE_ICON = "🏇"

# ==============================
# 頁面設定
# ==============================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# 應用標題
# ==============================
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("輸入賽馬相關特徵，AI 將預測最有可能獲勝的馬匹及其勝率分佈。")

# ==============================
# 特徵輸入表單
# ==============================
st.subheader("📊 輸入馬匹與騎師特徵")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("馬匹年齡 (歲)", min_value=2.0, max_value=12.0, value=4.5, step=0.5)
        win_rate = st.slider("過去勝率 (%)", 0, 100, 45) / 100.0
        jockey_rating = st.slider("騎師評分 (1-10)", 1, 10, 7)
    
    with col2:
        track_adapt = st.number_input("場地適應度 (0-10)", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
        recent_form = st.slider("近期表現指數 (0-10)", 0, 10, 8)
    
    # 提交按鈕
    submitted = st.form_submit_button("🔮 開始預測", type="primary", use_container_width=True)

# ==============================
# 處理預測請求
# ==============================
if submitted:
    # 組裝特徵向量（順序必須與模型訓練一致）
    features = [age, win_rate, jockey_rating, track_adapt, recent_form]
    
    with st.spinner("正在呼叫 AI 模型...（首次請求可能需 30–50 秒）"):
        try:
            # 發送 POST 請求
            response = requests.post(
                API_URL,
                json={"features": features},
                timeout=60  # 容忍 Render 冷啟動
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 顯示主要結果
                st.success(f"🏆 預測勝出馬匹: **{result['prediction']}**")
                
                # 勝率視覺化
                probs = result.get("probabilities", [])
                if len(probs) > 0:
                    st.subheader("📈 各馬匹勝率分佈")
                    horse_names = [f"馬匹_{i+1}" for i in range(len(probs))]
                    prob_dict = dict(zip(horse_names, probs))
                    st.bar_chart(prob_dict)
                
                # 除錯資訊（開發用）
                with st.expander("🔧 技術細節"):
                    st.write("**輸入特徵:**", features)
                    st.write("**完整回應:**")
                    st.json(result)
                    
            else:
                st.error(f"❌ API 回應錯誤 (狀態碼: {response.status_code})")
                st.code(response.text)
                
        except requests.exceptions.Timeout:
            st.error("⏰ 請求超時！可能是 Render 免費版正在冷啟動。")
            st.info("請等待 30–50 秒後再次點擊「開始預測」。")
        except requests.exceptions.ConnectionError:
            st.error("🌐 網路連線失敗，請檢查 API URL 是否正確。")
        except Exception as e:
            st.exception(f"⚠️ 發生未知錯誤: {str(e)}")

# ==============================
# 使用說明（摺疊區塊）
# ==============================
with st.expander("ℹ️ 使用說明與注意事項"):
    st.markdown("""
    ### 📌 功能說明
    - 此系統基於機器學習模型預測賽馬結果。
    - 所有預測僅供娛樂參考，不構成投注建議。
    
    ### ⏱️ 首次載入較慢？
    - 後端部署於 **Render 免費方案**，若 15 分鐘內無請求會自動休眠。
    - **首次點擊「開始預測」時，需等待 30–50 秒** 讓服務喚醒。
    - 喚醒後，後續請求將立即回應。
    
    ### 🔧 技術架構
    - **前端**: Streamlit（部署於 Streamlit Cloud）
    - **後端**: FastAPI（部署於 Render）
    - **通訊**: HTTPS POST 請求
    
    ### 📬 問題回報
    若持續無法使用，請檢查：
    1. 後端是否正常：[健康檢查](https://btr-horse-api.onrender.com/health)
    2. Swagger 文件：[API 文件](https://btr-horse-api.onrender.com/docs)
    """)

# ==============================
# 頁尾
# ==============================
st.divider()
st.caption("© 2025 馬匹預測系統 | Powered by FastAPI + Streamlit")
