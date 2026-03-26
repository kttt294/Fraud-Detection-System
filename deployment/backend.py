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
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# --- 0. KHỞI TẠO CƠ SỞ DỮ LIỆU ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        # 1. Bảng cho External API (Giao dịch thực tế)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                v_features JSONB,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'API (User App)',
                confirmed BOOLEAN DEFAULT FALSE
            )
        """)
        # 2. Bảng cho Dashboard (Phân tích nội bộ)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                v_features JSONB,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'Dashboard System',
                confirmed BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Để an toàn cho DB cũ, thêm cột nếu thiếu
        try:
            cur.execute("ALTER TABLE api_fraud_logs ADD COLUMN IF NOT EXISTS v_features JSONB;")
            cur.execute("ALTER TABLE api_fraud_logs ADD COLUMN IF NOT EXISTS confirmed BOOLEAN DEFAULT FALSE;")
            cur.execute("ALTER TABLE system_fraud_logs ADD COLUMN IF NOT EXISTS v_features JSONB;")
            cur.execute("ALTER TABLE system_fraud_logs ADD COLUMN IF NOT EXISTS confirmed BOOLEAN DEFAULT FALSE;")
        except Exception:
            pass # Bỏ qua nếu DB không hỗ trợ cú pháp này hoặc lỗi

        conn.commit()
        cur.close()
        conn.close()
        print("--- Database initialized with api_fraud_logs and system_fraud_logs tables ---")
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

# LOAD SCALER
SCALER_PATH = os.path.join(BASE_DIR, 'modeling', 'scaler.pkl')
scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"--- Scaler loaded successfully from {SCALER_PATH} ---")
else:
    print(f"Warning: Scaler Not Found at {SCALER_PATH}")

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
    transaction_time: str | None = None  # "HH:MM:SS" hoặc None (tự động lấy giờ server)
    v_features: list[float]
    source: str = "API (User App)"

def resolve_time_val(transaction_time: str | None) -> float:
    """Quy đổi giờ giao dịch sang giây trong ngày (0 -> 86400).
    - Nếu nhân viên cung cấp giờ (HH:MM:SS): dùng giờ đó.
    - Nếu là API tự động (None): dùng giờ server tại thời điểm nhận request.
    Đảm bảo time_val luôn nằm trong [0, 86400], đồng bộ hệ quy chiếu với model.
    """
    if transaction_time:
        try:
            parts = transaction_time.strip().split(":")
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
            return float(h * 3600 + m * 60 + s)
        except Exception:
            pass  # Nếu parse lỗi, fallback về giờ server
    now = datetime.now()
    return float(now.hour * 3600 + now.minute * 60 + now.second)

class BulkTransactions(BaseModel):
    transactions: list[Transaction]

@app.get("/")
def health_check():
    return {"status": "ONLINE", "model_loaded": model is not None}

@app.get("/alerts")
def get_alerts(limit: int = 10):
    """Lấy danh sách cảnh báo từ App người dùng (Thực tế)."""
    conn = get_db_connection()
    if not conn: return {"error": "DB_CONNECTION_FAILED"}
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Lấy thông tin bao gồm cả id và confirmed
        cur.execute("SELECT id, amount, fraud_probability, created_at, source, confirmed FROM api_fraud_logs ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close(); conn.close()
        return {"status": "success", "data": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ConfirmPayload(BaseModel):
    is_fraud: bool

@app.put("/confirm-fraud/{log_id}")
def confirm_fraud(log_id: int, payload: ConfirmPayload, is_system: bool = False):
    """API để Admin/người trực hệ thống xác nhận gian lận là thật hay giả."""
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB_CONNECTION_FAILED")
    try:
        cur = conn.cursor()
        target_table = "system_fraud_logs" if is_system else "api_fraud_logs"
        cur.execute(f"UPDATE {target_table} SET confirmed = %s WHERE id = %s", (payload.is_fraud, log_id))
        conn.commit()
        cur.close(); conn.close()
        return {"status": "success", "confirmed": payload.is_fraud, "id": log_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-bulk")
async def verify_bulk(payload: BulkTransactions):
    """Xử lý hàng loạt giao dịch từ file CSV (Hiệu suất cao)."""
    if not model:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa sẵn sàng.")
    
    try:
        # 1. Chuyển đổi và Scale dữ liệu
        if scaler:
            # Scale dùng RobustScaler (2 cột)
            X_raw = np.array([[tx.amount, tx.time_val] for tx in payload.transactions])
            X_scaled = scaler.transform(X_raw)
            # Ghép với V features
            v_data = np.array([tx.v_features for tx in payload.transactions])
            data_final = np.hstack([X_scaled, v_data])
        else:
            # Fallback
            data_final = [[tx.amount/100, tx.time_val/1000] + tx.v_features for tx in payload.transactions]
            
        df_batch = pd.DataFrame(data_final, columns=FEATURE_COLUMNS)
        
        # 2. AI Dự đoán đồng loạt
        probs = model.predict_proba(df_batch)[:, 1]
        
        # 3. Lọc lấy gian lận
        fraud_indices = np.where(probs > 0.5)[0]
        fraud_count = len(fraud_indices)
        
        fraud_list = []
        
        # 4. Lưu DB hàng loạt nếu có gian lận
        if fraud_count > 0:
            # Chuẩn bị dữ liệu trả về cho frontend
            for idx in fraud_indices:
                orig_tx = payload.transactions[idx]
                fraud_list.append({
                    "amount": orig_tx.amount,
                    "time_val": orig_tx.time_val,
                    "v_features": orig_tx.v_features,
                    "fraud_probability": float(probs[idx]),
                    "source": orig_tx.source
                })

            conn = get_db_connection()
            if conn:
                from psycopg2.extras import execute_values
                cur = conn.cursor()
                insert_data = [
                    (float(orig_tx["amount"]), float(orig_tx["time_val"]), json.dumps(orig_tx["v_features"]), float(orig_tx["fraud_probability"]), str(orig_tx["source"]))
                    for orig_tx in fraud_list
                ]
                
                # Lưu vào bảng SYSTEM kèm theo JSON v_features
                execute_values(cur, 
                    "INSERT INTO system_fraud_logs (amount, time_val, v_features, fraud_probability, source) VALUES %s", 
                    insert_data)
                conn.commit()
                cur.close(); conn.close()
                print(f"🚀 [BULK] Đã xử lý {len(payload.transactions)} dòng | Phát hiện & Lưu {fraud_count} gian lận.")

        return {
            "status": "success",
            "processed_count": len(payload.transactions),
            "fraud_detected": fraud_count,
            "frauds": fraud_list
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
        # Bước 1: Quy đổi giờ giao dịch -> giây trong ngày (0->86400)
        time_val = resolve_time_val(tx.transaction_time)

        # Bước 2: Scale Amount và Time
        if scaler:
            input_scaled = scaler.transform([[tx.amount, time_val]])
            scaled_amt = float(input_scaled[0][0])
            scaled_time = float(input_scaled[0][1])
        else:
            scaled_amt = tx.amount / 100
            scaled_time = time_val / 86400
            
        input_data = [scaled_amt, scaled_time] + tx.v_features
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        prob = float(model.predict_proba(input_df)[0][1])
        prediction = 1 if prob > 0.5 else 0
        
        # Chỉ lưu vào Dashboard nếu là giao dịch nghi vấn (Gian lận)
        if prediction == 1:
            conn = get_db_connection()
            if conn:
                # Phân loại bảng dựa trên nguồn gốc giao dịch
                # Đưa tất cả về bảng api_fraud_logs theo yêu cầu để hiện ở Giám sát Realtime
                target_table = "api_fraud_logs"
                
                cur = conn.cursor()
                cur.execute(
                    f"INSERT INTO {target_table} (amount, time_val, v_features, fraud_probability, source) VALUES (%s, %s, %s, %s, %s)",
                    (tx.amount, time_val, json.dumps(tx.v_features), prob, tx.source)
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
