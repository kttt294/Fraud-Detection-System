from fastapi import FastAPI, HTTPException
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

# Tự động xác định thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load biến môi trường từ .env
load_dotenv(os.path.join(BASE_DIR, '.env'))

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="SafeGuard Banking Real-time API",
    description="Hệ thống lõi xử lý giao dịch tự động của ngân hàng",
    version="1.0.0"
)

# 2. Cấu hình mô hình
MODEL_PATH = os.path.join(BASE_DIR, 'modeling', 'model.pkl')
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

# 3. Load model khi khởi động
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

def get_db_connection():
    try:
        # Cấu hình SSL linh hoạt
        ssl_args = {"sslmode": os.getenv("DB_SSLMODE", "require")}
        ca_path = os.path.join(BASE_DIR, os.getenv("DB_CA_PATH") or "deployment/certs/ca.pem")
        
        if os.path.exists(ca_path):
            ssl_args["sslrootcert"] = ca_path
            print(f"[INFO] Đang sử dụng chứng chỉ CA tại: {ca_path}")
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

# Tạo bảng nếu chưa tồn tại
@app.on_event("startup")
async def startup_event():
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                transaction_id TEXT UNIQUE,
                timestamp TIMESTAMP,
                amount FLOAT,
                decision TEXT,
                fraud_probability TEXT,
                processing_time_ms TEXT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("--- Database initialized ---")

# 5. Định nghĩa cấu trúc dữ liệu gửi đến (Schema)
class Transaction(BaseModel):
    amount: float = Field(..., example=150.75)
    time_offset: float = Field(..., example=1200.0)
    v_features: list[float] = Field(..., min_items=28, max_items=28)

class BatchTransactions(BaseModel):
    transactions: list[Transaction]

@app.get("/")
def home():
    return {"status": "ONLINE", "message": "Ngân hàng SafeGuard - Hệ thống lõi đã sẵn sàng."}

@app.get("/logs")
def get_logs():
    """Lấy 5 giao dịch gần nhất từ Database cho Dashboard"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT timestamp, transaction_id, decision, amount, fraud_probability FROM transactions ORDER BY id DESC LIMIT 5")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return {"status": "success", "data": rows}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "DB_CONNECTION_FAILED"}

@app.post("/verify")
async def verify_transaction(tx: Transaction):
    """
    API tiếp nhận giao dịch trực tiếp từ Máy POS, Web, App.
    Trả về kết quả Chấp thuận hoặc Từ chối trong mili giây.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được tải lên máy chủ.")
    
    start_time = time.time()
    
    try:
        # Chuẩn bị DataFrame đúng thứ tự mô hình yêu cầu
        input_data = [tx.amount/100, tx.time_offset/1000] + tx.v_features
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # Dự đoán
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        process_time = time.time() - start_time
        
        # Logic nghiệp vụ: Nếu gian lận xác suất cao -> Từ chối ngay
        decision = "BLOCK" if prediction == 1 else "APPROVE"
        
        result = {
            "transaction_id": f"TXN-{int(time.time()*1000)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": tx.amount,
            "decision": decision,
            "fraud_probability": f"{probability:.4%}",
            "processing_time_ms": f"{process_time*1000:.2f}ms"
        }

        # --- GHI VÀO POSTGRESQL (Aiven) ---
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO transactions (transaction_id, timestamp, amount, decision, fraud_probability, processing_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (result['transaction_id'], result['timestamp'], result['amount'], result['decision'], result['fraud_probability'], result['processing_time_ms']))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as db_err:
                print(f"Lỗi khi ghi vào DB: {db_err}")

        # --- GHI VÀO JSON (Dự phòng) ---
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
        
        # Dự đoán hàng loạt
        predictions = model.predict(input_df)
        # Chuyển thành text để trả về
        results = ["GIAN LẬN" if p == 1 else "Hợp lệ" for p in predictions]
        
        return {"status": "success", "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi batch: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
