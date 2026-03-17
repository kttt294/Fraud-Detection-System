from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os
import time
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- 0. KHỞI TẠO CƠ SỞ DỮ LIỆU ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        # Tạo bảng logs thống nhất cho cả Dashboard và API
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'HỆ THỐNG'
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("--- Database initialized with fraud_logs table ---")
    yield

app = FastAPI(
    title="SafeGuard Banking Core API",
    description="Hệ thống lõi xử lý giao dịch và phát hiện gian lận",
    version="2.0.0",
    lifespan=lifespan
)

# --- 1. CONFIG & MODEL ---
MODEL_PATH = os.path.join(BASE_DIR, 'modeling', 'model.pkl')
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"--- Model loaded successfully from {MODEL_PATH} ---")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE", "require")
        )
        return conn
    except Exception as e:
        print(f"Connection error: {e}")
        return None

class Transaction(BaseModel):
    amount: float
    time_val: float
    v_features: list[float]
    source: str = "API (User App)"

class BulkTransactions(BaseModel):
    transactions: list[Transaction]

@app.get("/")
def health_check():
    return {"status": "ONLINE", "model_loaded": model is not None}

@app.get("/alerts")
def get_alerts(limit: int = 10):
    """Lấy danh sách cảnh báo mới nhất cho Dashboard."""
    conn = get_db_connection()
    if not conn: return {"error": "DB_CONNECTION_FAILED"}
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT amount, fraud_probability, created_at, source FROM fraud_logs ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close(); conn.close()
        return {"status": "success", "data": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/verify-bulk")
async def verify_bulk(payload: BulkTransactions):
    """Xử lý hàng loạt giao dịch từ file CSV (Hiệu suất cao)."""
    if not model:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa sẵn sàng.")
    
    try:
        # 1. Chuyển đổi list objects sang DataFrame để xử lý mảng
        data_list = []
        for tx in payload.transactions:
            row = [tx.amount/100, tx.time_val/1000] + tx.v_features
            data_list.append(row)
        
        df_batch = pd.DataFrame(data_list, columns=FEATURE_COLUMNS)
        
        # 2. AI Dự đoán đồng loạt
        # Sử dụng .values để tránh UserWarning về tên cột
        probs = model.predict_proba(df_batch.values)[:, 1]
        
        # 3. Lọc lấy gian lận
        fraud_indices = np.where(probs > 0.5)[0]
        fraud_count = len(fraud_indices)
        
        # 4. Lưu DB hàng loạt nếu có gian lận
        if fraud_count > 0:
            conn = get_db_connection()
            if conn:
                from psycopg2.extras import execute_values
                cur = conn.cursor()
                insert_data = []
                for idx in fraud_indices:
                    tx = payload.transactions[idx]
                    insert_data.append((tx.amount, tx.time_val, float(probs[idx]), tx.source))
                
                execute_values(cur, 
                    "INSERT INTO fraud_logs (amount, time_val, fraud_probability, source) VALUES %s", 
                    insert_data)
                conn.commit()
                cur.close(); conn.close()
                print(f"🚀 [BULK] Đã xử lý {len(payload.transactions)} dòng | Phát hiện & Lưu {fraud_count} gian lận.")

        return {
            "status": "success",
            "processed_count": len(payload.transactions),
            "fraud_detected": fraud_count
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify_transaction(tx: Transaction):
    """Endpoint chính tiếp nhận và đánh giá giao dịch."""
    if not model:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa sẵn sàng.")
    
    try:
        # Chuẩn bị dữ liệu (Standardizing logic)
        input_data = [tx.amount/100, tx.time_val/1000] + tx.v_features
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        prob = float(model.predict_proba(input_df)[0][1])
        prediction = 1 if prob > 0.5 else 0
        
        # Chỉ lưu vào Dashboard nếu là giao dịch nghi vấn (Gian lận)
        if prediction == 1:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO fraud_logs (amount, time_val, fraud_probability, source) VALUES (%s, %s, %s, %s)",
                    (tx.amount, tx.time_val, prob, tx.source)
                )
                conn.commit()
                cur.close(); conn.close()
 
        return {
            "decision": "BLOCK" if prediction == 1 else "APPROVE",
            "probability": f"{prob:.2%}",
            "source_recorded": tx.source if prediction == 1 else "None"
        }
    except Exception as e:
        print(f"💥 Backend Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
