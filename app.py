import streamlit as st
import requests
import json

# ======================
# 設定頁面
# ======================
st.set_page_config(
    page_title="🏇 賽馬入位預測系統",
    page_icon="🏇",
    layout="centered"
)

st.title("🏇 賽馬入位預測系統")
st.markdown("輸入馬匹資料，AI 將預測該馬是否能進入前三名（入位）")

# ======================
# API 設定
# ======================
API_URL = "https://btr-horse-api.onrender.com/predict"

# ======================
# 初始化 session state
# ======================
if 'history' not in st.session_state:
    st.session_state.history = []

# ======================
# 使用者輸入表單
# ======================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        actual_weight = st.number_input("實際負重 (kg)", min_value=0.0, max_value=100.0, value=56.0, step=0.5)
        draw = st.number_input("檔位", min_value=1, max_value=14, value=3)
        jockey_id = st.number_input("騎師 ID", min_value=1, value=1)
    
    with col2:
        declared_weight = st.number_input("宣告負重 (kg)", min_value=0.0, max_value=100.0, value=58.0, step=0.5)
        win_odds = st.number_input("獨贏賠率", min_value=1.0, value=3.5, step=0.1)
        trainer_id = st.number_input("練馬師 ID", min_value=1, value=2)
    
    submitted = st.form_submit_button("🎯 預測")

# ======================
# 處理預測請求
# ======================
if submitted:
    # 構建特徵列表（順序必須與訓練一致！）
    payload = {
        "features": [
            float(actual_weight),
            float(declared_weight),
            int(draw),
            float(win_odds),
            int(jockey_id),
            int(trainer_id)
        ]
    }
    
    # 顯示除錯資訊（上線後可註解掉）
    # st.write("📤 發送至 API 的資料:", payload)
    
    try:
        with st.spinner("🧠 正在連接 AI 模型，請稍候..."):
            response = requests.post(
                API_URL,
                json=payload,  # 關鍵：使用 json= 自動設定 Content-Type
                timeout=15
            )
        
        # 顯示除錯資訊（上線後可註解掉）
        # st.write("📡 API 回應狀態碼:", response.status_code)
        # st.write("📨 原始回應內容:", response.text)
        
        if response.status_code == 200:
            result = response.json()
            
            # 顯示結果
            prob = result["top3_probability"]
            prediction = result["prediction"]
            
            if prediction == "入位":
                st.success(f"🟢 **預測結果：入位**（機率：{prob:.1%}）")
            else:
                st.error(f"🔴 **預測結果：未入位**（機率：{1 - prob:.1%}）")
            
            # 儲存歷史紀錄
            record = {
                "input": {
                    "actual_weight": actual_weight,
                    "declared_weight": declared_weight,
                    "draw": draw,
                    "win_odds": win_odds,
                    "jockey_id": jockey_id,
                    "trainer_id": trainer_id
                },
                "result": result
            }
            st.session_state.history.insert(0, record)
            
        else:
            st.error(f"❌ API 回應錯誤：{response.status_code}")
            st.code(response.text)
            
    except requests.exceptions.Timeout:
        st.error("⏰ 請求超時！請檢查網路或稍後再試。")
    except requests.exceptions.ConnectionError:
        st.error("🔌 無法連接到 AI 伺服器！請確認 URL 是否正確。")
    except Exception as e:
        st.error(f"💥 發生未知錯誤：{str(e)}")

# ======================
# 顯示歷史紀錄
# ======================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 預測歷史")
    
    for i, record in enumerate(st.session_state.history[:5]):  # 只顯示最近 5 筆
        input_data = record["input"]
        result = record["result"]
        
        with st.expander(f"預測 #{len(st.session_state.history) - i}"):
            st.write("**輸入資料**")
            st.json(input_data)
            st.write("**預測結果**")
            st.json({
                "prediction": result["prediction"],
                "top3_probability": f"{result['top3_probability']:.4f}"
            })

# ======================
# 健康檢查（可選）
# ======================
with st.sidebar:
    st.header("🔧 系統狀態")
    try:
        health_resp = requests.get("https://btr-horse-api.onrender.com/health", timeout=5)
        if health_resp.status_code == 200:
            st.success("✅ AI 伺服器正常運作！")
        else:
            st.warning("⚠️ AI 伺服器異常")
    except:
        st.error("❌ 無法連接 AI 伺服器")
