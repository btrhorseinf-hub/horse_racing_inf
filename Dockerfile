FROM python:3.11-slim

WORKDIR /app

# 安裝相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼
COPY . .

# 啟動 FastAPI（注意 port=10000）
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]
