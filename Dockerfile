# Dockerfile —— 賽馬預測系統（支援 shap 安裝）

FROM python:3.12-slim

WORKDIR /app

# 安裝編譯 shap 所需的系統依賴（關鍵！）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 暴露 Streamlit 埠
EXPOSE 8501

# 啟動命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
