import streamlit as st
import requests
import json

# ==============================
# 配置
# ==============================
API_URL = "https://btr-horse-api.onrender.com/predict"
PAGE_TITLE = "🏇 馬匹勝率預測系統"
PAGE_ICON = "🏇"

# ==============================
# 頁面設定
# ==============================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# ==============================
# 應用標題與說明
# ==============================
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("""
輸入馬匹與騎師的相關特徵，AI 將預測最有可能獲勝的馬匹及其勝率分佈。
""")

# ==============================
# 特徵輸入（根據你的模型調整）
# ==============================
st.subheader("📊 輸入特徵")

# 假設你的模型需要 5 個數值特徵（請依實際情況修改）
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("馬匹年齡 (歲)", min_value=2.0, max_value=10.0, value=4.5, step=0.5)
    win_rate = st.slider("過去勝率 (%)", 0, 100, 45) / 100.0  # 轉為 0~1
    jockey_rating = st.slider("騎師評分 (1-10)", 1, 10, 7)

with col2:
    track_adapt = st.number_input("場地適應度 (0-10)", min_value=0.0, max_value=10.0, value=6.5, step=0.5)
    recent_form = st.slider("近期表現指數", 0, 10, 8)

# 組裝特徵向量（順序必須與訓練時一致！）
features = [age, win_rate, jockey_rating, track_adapt, recent_form]

st.info(f"輸入特徵向量: {features}")

# ==============================
# 預測按鈕
# ==============================
if st.button("🔮 開始預測", type="primary", use_container_width=True):
    with st.spinner("正在呼叫 AI 模型..."):
        try:
            # 發送 POST 請求到你的 Render API
            response = requests.post(
                API_URL,
                json={"features": features},
                timeout=30  # 避免免費版冷啟動超時
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 顯示結果
                st.success(f"🏆 預測勝出馬匹: **{result['prediction']}**")
                
                # 顯示勝率分佈（假設有 5 匹馬）
                st.subheader("📈 各馬匹勝率分佈")
                probs = result.get("probabilities", [])
                horse_names = [f"馬匹_{i+1}" for i in range(len(probs))] if probs else []
                
                if probs and horse_names:
                    prob_data = dict(zip(horse_names, probs))
                    st.bar_chart(prob_data)
                else:
                    st.write("無法顯示勝率圖表（資料格式不符）")
                
                # 除錯資訊（可選）
                with st.expander("🔧 除錯資訊"):
                    st.json(result)
                    
            else:
                st.error(f"❌ API 回應錯誤 (狀態碼: {response.status_code})")
                st.code(response.text)
                
        except requests.exceptions.Timeout:
            st.error("⏰ 請求超時！可能是 Render 免費版正在冷啟動，請稍等 30 秒後重試。")
        except requests.exceptions.ConnectionError:
            st.error("🌐 網路連線失敗，請檢查 URL 或網路。")
        except Exception as e:
            st.exception(f"⚠️ 發生未知錯誤: {str(e)}")

# ==============================
# 使用說明
# ==============================
with st.expander("ℹ️ 使用說明"):
    st.markdown("""
    - 此系統基於機器學習模型預測賽馬結果。
    - 所有預測僅供娛樂參考，不構成投注建議。
    - 若首次點擊「開始預測」無反應，請等待 30–50 秒（Render 免費版冷啟動）。
    - API 由 [FastAPI](https://fastapi.tiangolo.com/) 提供，部署於 [Render](https://render.com/)。
    """)

# ==============================
# 頁尾
# ==============================
st.divider()
st.caption("© 2025 馬匹預測系統 | Powered by FastAPI + Streamlit")
