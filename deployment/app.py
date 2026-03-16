# File này phục vụ việc demo ứng dụng thông qua deploy lên Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import json
import psycopg2
from dotenv import load_dotenv

# --- 1. CẤU HÌNH BAN ĐẦU ---
st.set_page_config(
    page_title="SafeGuard Banking | Fraud Detection",
    page_icon=None,
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- 2. LOAD MODEL (Sử dụng Cache để tối ưu) ---
MODEL_PATH = os.path.join(BASE_DIR, 'modeling', 'model.pkl')
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# --- 3. KẾT NỐI DATABASE ---
def get_db_connection():
    try:
        ssl_args = {"sslmode": os.getenv("DB_SSLMODE", "require")}
        ca_path = os.path.join(BASE_DIR, os.getenv("DB_CA_PATH") or "deployment/certs/ca.pem")
        
        if os.path.exists(ca_path):
            ssl_args["sslrootcert"] = ca_path
        
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            **ssl_args
        )
        return conn
    except:
        return None

# --- 4. LOGIC NGHIỆP VỤ ---
def process_prediction(amount, time_offset, v_features):
    if model is None:
        return None
        
    input_data = [amount/100, time_offset/1000] + v_features
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    decision = "BLOCK" if prediction == 1 else "APPROVE"
    
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "amount": amount,
        "decision": decision,
        "fraud_probability": f"{probability:.4%}"
    }
    
    # Lưu vào DB nếu bị BLOCK
    if decision == "BLOCK":
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
            except:
                pass
    return result

# --- 5. GIAO DIỆN CHÍNH ---
st.title("SafeGuard Banking | Fraud Detection", anchor=False)

# Sidebar
st.sidebar.header("Operations Menu", anchor=False)
menu = st.sidebar.radio(
    "Select Operation Mode:", 
    ["Transaction Investigation", "Periodic Batch Audit"]
)

if menu == "Transaction Investigation":
    st.subheader("Kiểm Tra Giao Dịch Thủ Công", anchor=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        v17 = st.number_input("V17", value=0.0)
        v14 = st.number_input("V14", value=0.0)
    with col2:
        v16 = st.number_input("V16", value=0.0)
        v12 = st.number_input("V12", value=0.0)
    with col3:
        amount = st.number_input("Amount (Euro)", value=100.0)
        time_val = st.number_input("Time (s)", value=1000.0)
        
    if st.button("Kiểm tra ngay"):
        v_features = [0.0] * 28
        for idx, val in {17: v17, 14: v14, 16: v16, 12: v12}.items():
            v_features[idx-1] = val
            
        res = process_prediction(amount, time_val, v_features)
            
        if res:
            if res['decision'] == "BLOCK":
                st.error(f"CẢNH BÁO: Hệ thống CHẶN giao dịch! (Xác suất gian lận: {res['fraud_probability']})")
            else:
                st.success(f"Giao dịch ĐƯỢC CHẤP THUẬN. (Xác suất: {res['fraud_probability']})")
        else:
            st.warning("Lỗi: Không thể tải mô hình dự đoán.")

elif menu == "Periodic Batch Audit":
    st.subheader("Kiểm Tra Giao Dịch Hàng Loạt", anchor=False)
    uploaded_file = st.file_uploader("Chọn file CSV giao dịch", type="csv")
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Xem trước dữ liệu:", df_upload.head())
        
        if st.button("Tiến hành phân tích toàn bộ"):
            if model is None:
                st.error("Model chưa được tải.")
            else:
                with st.spinner('Đang đẩy dữ liệu lên Backend xử lý...'):
                    # Chuẩn bị dữ liệu
                    data_list = []
                    for _, row in df_upload.iterrows():
                        v_feats = [float(row[f'V{i}']) for i in range(1, 29)]
                        data_list.append([row['Amount']/100 if 'Amount' in row else 0, 
                                        row['Time']/1000 if 'Time' in row else 0] + v_feats)
                    
                    input_df = pd.DataFrame(data_list, columns=FEATURE_COLUMNS)
                    predictions = model.predict(input_df)
                    df_upload['Kết quả'] = ["GIAN LẬN" if p == 1 else "Hợp lệ" for p in predictions]
                    
                    st.success("Phân tích hoàn tất!")
                    st.dataframe(df_upload)

st.divider()
