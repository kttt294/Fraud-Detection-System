from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os
import time
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load biến môi trường từ .env
load_dotenv(os.path.join(BASE_DIR, '.env'))

# 0. Khởi tạo Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS suspicious_activities (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                amount FLOAT,
                decision TEXT,
                fraud_probability TEXT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("--- Database initialized---")
    yield

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="SafeGuard Banking Real-time API",
    description="Hệ thống lõi xử lý giao dịch tự động của ngân hàng",
    version="1.0.0",
    lifespan=lifespan
)

# 2. Cấu hình mô hình
MODEL_PATH = os.path.join(BASE_DIR, 'modeling', 'model.pkl')
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

# 3. Load model khi khởi động
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# 4. Hàm kết nối Database
def get_db_connection():
    try:
        # Cấu hình SSL cho kết nối PostgreSQL
        ssl_args = {"sslmode": os.getenv("DB_SSLMODE", "require")}
        ca_path = os.path.join(BASE_DIR, os.getenv("DB_CA_PATH") or "deployment/certs/ca.pem")
        
        if os.path.exists(ca_path):
            ssl_args["sslrootcert"] = ca_path
            print(f"[INFO] Đang sử dụng chứng chỉ xác thực CA tại: {ca_path}")
        else:
            print("[WARNING] Không tìm thấy file ca.pem. Cố gắng kết nối không có chứng chỉ xác thực...")

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            **ssl_args
        )
        return conn
    except Exception as e:
        print(f"Lỗi kết nối DB: {e}")
        return None

# 5. Định nghĩa khung dữ liệu đầu vào (API Schemas)
class Transaction(BaseModel):
    amount: float = Field(...)
    time_offset: float = Field(...)
    v_features: list[float] = Field(..., min_items=28, max_items=28)

class BatchTransactions(BaseModel):
    transactions: list[Transaction]

@app.get("/")
def home():
    return {"status": "ONLINE", "message": "Ngân hàng SafeGuard - Hệ thống lõi đã sẵn sàng."}

@app.get("/logs")
def get_logs():
    """Lấy 5 giao dịch nghi vấn gần nhất từ Database."""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT id, timestamp, decision, amount, fraud_probability FROM suspicious_activities ORDER BY id DESC LIMIT 5")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return {"status": "success", "data": rows}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "DB_CONNECTION_FAILED"}

@app.post("/verify")
async def verify_transaction(tx: Transaction):
    """API tiếp nhận giao dịch trực tiếp từ Máy POS, Web, App."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải lên máy chủ.")
    
    try:
        # Chuẩn bị DataFrame đúng thứ tự mô hình yêu cầu
        input_data = [tx.amount/100, tx.time_offset/1000] + tx.v_features
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # Dự đoán
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Logic nghiệp vụ: Nếu gian lận xác suất cao -> Từ chối ngay
        decision = "BLOCK" if prediction == 1 else "APPROVE"
        
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": tx.amount,
            "decision": decision,
            "fraud_probability": f"{probability:.4%}"
        }

        # --- CHỈ LƯU VẾT NẾU GIAO DỊCH BỊ NGHI VẤN (BLOCK) ---
        if decision == "BLOCK":
            # 1. Ghi vào PostgreSQL
            conn = get_db_connection()
            if conn:
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO suspicious_activities (timestamp, amount, decision, fraud_probability)
                        VALUES (%s, %s, %s, %s)
                    """, (result['timestamp'], result['amount'], result['decision'], result['fraud_probability']))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as db_err:
                    print(f"Lỗi khi ghi vào DB: {db_err}")

            # 2. Ghi vào JSON (Dự phòng)
            log_file = os.path.join(BASE_DIR, 'data', 'outputs', 'logs.json')
            logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    try:
                        logs = json.load(f)
                    except:
                        logs = []
            
            logs.append(result)
            with open(log_file, 'w') as f:
                json.dump(logs[-100:], f, indent=4)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi xử lý dữ liệu: {str(e)}")

@app.post("/verify-batch")
async def verify_batch(batch: BatchTransactions):
    """Xử lý đồng thời nhiều giao dịch (Dành cho Batch Audit)"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải.")
    
    try:
        # Chuyển đổi toàn bộ batch thành DataFrame
        data_list = []
        for tx in batch.transactions:
            data_list.append([tx.amount/100, tx.time_offset/1000] + tx.v_features)
        
        input_df = pd.DataFrame(data_list, columns=FEATURE_COLUMNS)
        
        predictions = model.predict(input_df)
        results = ["GIAN LẬN" if p == 1 else "Hợp lệ" for p in predictions]
        
        return {"status": "success", "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi batch: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
