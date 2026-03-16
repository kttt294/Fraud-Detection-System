import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import requests

# Cấu hình trang
st.set_page_config(
    page_title="SafeGuard Banking | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_BASE_URL = "http://localhost:8000"

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ SafeGuard Banking | Fraud Analyst Operations Dashboard")
st.markdown("""
Welcome back, **Analyst**. Current System Status: <span style='color:green; font-weight:bold'>ACTIVE</span> | 
Architecture: `Pure-Frontend (Decoupled)` | Backend: `FastAPI`
""", unsafe_allow_html=True)

# Danh sách cột bắt buộc
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

# --- Sidebar: Chọn dịch vụ ---
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png")
st.sidebar.header("Operations Menu")
menu = st.sidebar.selectbox(
    "Select Operation Mode:", 
    ["🔍 Transaction Investigation", "📊 Periodic Batch Audit"]
)

if menu == "🔍 Transaction Investigation":
    st.subheader("🕵️ Deep Investigation Mode")
    st.info("Dành cho chuyên viên điều tra các giao dịch bị hệ thống Core-Banking đánh dấu nghi vấn.")
    
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
        
    if st.button("🚀 Kiểm tra ngay"):
        v_features = [0.0] * 28
        v_map = {17: v17, 14: v14, 16: v16, 12: v12}
        for i, val in v_map.items():
            v_features[i-1] = val
            
        payload = {"amount": amount, "time_offset": time_val, "v_features": v_features}
        
        try:
            res = requests.post(f"{API_BASE_URL}/verify", json=payload, timeout=5).json()
            if res['decision'] == "BLOCK":
                st.error(f"⚠️ **CẢNH BÁO:** Hệ thống CHẶN giao dịch! (Xác suất gian lận: {res['fraud_probability']})")
                st.image("https://img.icons8.com/color/96/000000/high-priority.png")
            else:
                st.success(f"✅ Giao dịch ĐƯỢC CHẤP THUẬN. (Xác suất: {res['fraud_probability']})")
                st.image("https://img.icons8.com/color/96/000000/verified-badge.png")
        except:
            st.error("❌ Không thể kết nối tới Backend API (Port 8000).")

elif menu == "📊 Periodic Batch Audit":
    st.subheader("📂 Batch Audit Interface")
    uploaded_file = st.file_uploader("Chọn file CSV giao dịch", type="csv")
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Xem trước dữ liệu:", df_upload.head())
        
        if st.button("🔍 Tiến hành phân tích File qua API"):
            # Chuyển đổi DataFrame thành format API yêu cầu
            transactions = []
            for _, row in df_upload.iterrows():
                v_feats = [float(row[f'V{i}']) for i in range(1, 29)]
                transactions.append({
                    "amount": float(row['Amount']) if 'Amount' in row else 0.0,
                    "time_offset": float(row['Time']) if 'Time' in row else 0.0,
                    "v_features": v_feats
                })
            
            payload = {"transactions": transactions}
            
            try:
                with st.spinner('Đang đẩy dữ liệu lên Backend xử lý...'):
                    response = requests.post(f"{API_BASE_URL}/verify-batch", json=payload)
                    if response.status_code == 200:
                        results = response.json()['predictions']
                        df_upload['Kết quả'] = results
                        st.success("✅ Phân tích hoàn tất!")
                        st.dataframe(df_upload)
                    else:
                        st.error(f"Lỗi API: {response.text}")
            except Exception as e:
                st.error(f"Lỗi kết nối: {str(e)}")

# --- THEO DÕI LIVE (Gọi từ API) ---
st.sidebar.divider()
st.sidebar.subheader("Live Monitor (via API)")

try:
    log_res = requests.get(f"{API_BASE_URL}/logs", timeout=3).json()
    if log_res['status'] == "success":
        for log in log_res['data']:
            color = "red" if log['decision'] == "BLOCK" else "green"
            st.sidebar.markdown(f"**{log['timestamp']}**")
            st.sidebar.markdown(f"ID: `{log['transaction_id']}`")
            st.sidebar.markdown(f"Status: <span style='color:{color}'>{log['decision']}</span> ({log['amount']}€)", unsafe_allow_html=True)
            st.sidebar.divider()
    else:
        st.sidebar.info("Đang đợi dữ liệu từ Backend...")
except:
    st.sidebar.warning("⚠️ Không thể lấy log từ API.")

if st.sidebar.button("🔄 Làm mới dữ liệu"):
    st.rerun()

# --- Footer ---
st.divider()
st.caption("© 2024 SafeGuard Banking Intelligence | Đội ngũ Phân tích Rủi ro")
