"""
File phục vụ mục đích Deploy ứng dụng Fraud Detection lên Streamlit Cloud.
Dashboard quản trị: Theo dõi API Real-time & Phân tích Giao dịch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import time
import requests
from datetime import datetime

# --- 1. CONFIG & SETUP ---
load_dotenv()
st.set_page_config(page_title="SafeGuard Banking - Credit Card Fraud Detection", layout="wide")

# NHÚNG TOÀN BỘ CSS
CSS_EMBEDDED = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: white !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    padding: 60px 40px 20px 40px !important; /* Giảm padding top để kéo body lên */
    max-width: 1400px !important;
    margin: 0 auto !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    display: none !important;
}

.custom-header {
    position: fixed;
    top: 0; left: 0; width: 100%;
    height: 60px;
    background-color: #38bdf8;
    display: flex; align-items: center;
    padding: 0 40px; z-index: 10000;
    color: white; justify-content: space-between;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.header-branding { font-weight: 700; font-size: 1.2rem; display: flex; align-items: center; }
.header-right { display: flex; align-items: center; gap: 25px; }
.header-icons { display: flex; gap: 20px; align-items: center; }
.icon-wrapper { position: relative; cursor: pointer; }

.icon-dot {
    position: absolute;
    top: -2px; right: -2px;
    width: 8px; height: 8px;
    background-color: #0ea5e9;
    border: 1.5px solid #38bdf8;
    border-radius: 50%;
}

.user-block {
    display: flex;
    align-items: center;
    gap: 12px;
}

.user-info {
    line-height: 1.2;
    font-size: 0.9rem;
}

.user-name { font-weight: 700; }
.user-role { font-size: 0.75rem; opacity: 0.9; }

.vertical-divider {
    border-left: 1px solid #e2e8f0;
    height: 400px;
    margin: 20px auto;
    width: 0;
}

.section-header {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    display: flex;
    align-items: center;
    margin-bottom: 20px !important;
}

.live-monitor-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    display: flex;
    align-items: center;
    margin-bottom: 20px !important;
}

.live-dot {
    height: 8px; width: 8px; background-color: #ef4444;
    border-radius: 50%; display: inline-block; margin-right: 10px;
    animation: blink 1.5s infinite;
}
@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }

.alert-card {
    padding: 12px; background-color: #f8fafc;
    border-radius: 8px; margin-bottom: 15px;
    border: 1px solid #f1f5f9;
}
.alert-source { font-size: 0.65rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
.alert-meta { font-size: 0.75rem; color: #94a3b8; margin-top: 4px; }

[data-testid="stNumberInput"] {
    max-width: 160px !important;
    margin-bottom: 10px !important;
    margin-top: -20px !important;
}

.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button[data-testid="stBaseButton-primary"] {
    min-width: 200px !important; 
    background-color: #38bdf8 !important;
    color: white !important; border-radius: 6px !important;
    border: none !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* Thu nhỏ và căn giữa vùng tải file */
[data-testid="stFileUploader"] {
    width: 70% !important;
    margin: 0 auto !important;
}

.stButton > button:active {
    transform: scale(0.98) !important;
}

.v-feature-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 28px;
    margin-bottom: 0;
}

.feature-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: #1e293b;
    line-height: 28px;
}

/* Chỉ tác động lên nút X (secondary) trong bảng nhập liệu, không ảnh hưởng nút chính (primary) */
.stButton:has(> button[data-testid="stBaseButton-secondary"]) {
    margin-top: -28px !important;
    margin-bottom: 0 !important;
    justify-content: flex-end !important;
    padding: 0 !important;
    max-width: 160px !important;
}

.center-btn {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.stButton > button[data-testid="stBaseButton-secondary"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #94a3b8 !important;
    font-size: 1rem !important;
    padding: 0 !important;
    min-width: unset !important;
    width: 24px !important;
    height: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}

.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    color: #ef4444 !important;
    transform: scale(1.1) !important;
}

span[data-baseweb="tag"], button[aria-label="Clear all"] {
    display: none !important;
}

div[data-testid="stNumberInput"] button {
    display: none !important;
}
</style>
"""
st.markdown(CSS_EMBEDDED, unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
FEATURE_COLUMNS = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]

@st.cache_resource
def load_model():
    model_path = 'modeling/model.pkl'
    model_url = "https://github.com/kttt294/SafeGuard-Fraud-Detection-System/releases/download/v1.0.0/model.pkl"
    
    try:
        if not os.path.exists(model_path):
            if not model_url:
                st.error("Chưa cấu hình MODEL_URL trong Secrets! Không thể tải mô hình AI.")
                return None
            
            os.makedirs('modeling', exist_ok=True)
            with st.spinner("Đang tải mô hình AI từ GitHub Release (chỉ thực hiện lần đầu)..."):
                response = requests.get(model_url, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error(f"Lỗi khi tải model: HTTP {response.status_code}")
                    return None

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi khởi tạo mô hình: {e}")
    return None

model = load_model()

@st.cache_resource
def get_db_pool():
    from psycopg2 import pool
    try:
        host = os.getenv("DB_HOST")
        if not host: return None
        ssl_mode = os.getenv("DB_SSLMODE", "require")
        ca_path = os.getenv("DB_CA_PATH", "deployment/certs/ca.pem")
        
        params = {
            "host": host,
            "port": os.getenv("DB_PORT"),
            "database": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "sslmode": ssl_mode
        }
        
        # Nếu có file certificate thì mới dùng sslrootcert
        if os.path.exists(ca_path) and ssl_mode == "require":
            params["sslrootcert"] = ca_path
        elif ssl_mode == "require":
            # Nếu yêu cầu require nhưng không tìm thấy file CA, hạ cấp xuống allow để tránh lỗi treo app
            params["sslmode"] = "allow"

        return pool.SimpleConnectionPool(1, 20, **params)
    except: return None

@st.cache_resource
def init_db_cloud():
    conn = None
    try:
        conn = get_db_pool().getconn()
        cur = conn.cursor()
        # 1. Bảng API
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'API (User App)'
            )
        """)
        # 2. Bảng System
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'Dashboard System'
            )
        """)
        conn.commit()
        cur.close()
    except: pass
    finally:
        if conn: release_db_connection(conn)

def get_db_connection():
    if db_pool: return db_pool.getconn()
    return None

def release_db_connection(conn):
    if db_pool and conn: db_pool.putconn(conn)

db_pool = get_db_pool()
if db_pool: init_db_cloud()

def get_api_alerts():
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # Chỉ lấy log từ bảng API (Giao dịch thực tế)
            cur.execute("SELECT amount, fraud_probability, created_at, source FROM api_fraud_logs ORDER BY created_at DESC LIMIT 8")
            rows = cur.fetchall()
            cur.close()
            return rows
    except: pass
    finally:
        if conn: release_db_connection(conn)
    return []

def process_prediction(amount, time_val, v_features, source="HỆ THỐNG (Manual)"):
    if model:
        conn = None
        try:
            input_data = [amount/100, time_val/1000] + v_features
            features_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
            prob = model.predict_proba(features_df)[0][1]
            decision = "BLOCK" if prob > 0.5 else "APPROVE"
            
            if decision == "BLOCK":
                conn = get_db_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("INSERT INTO system_fraud_logs (amount, time_val, fraud_probability, source) VALUES (%s, %s, %s, %s)", (amount, time_val, prob, source))
                    conn.commit()
                    cur.close()
            return {"decision": decision, "prob": f"{prob:.2%}"}
        except: pass
        finally:
            if conn: release_db_connection(conn)
    return None

def process_bulk_cloud(df, amt_col, time_col, source="HỆ THỐNG (Bulk)"):
    if not model: return 0
    
    # 1. Chuẩn bị mảng dữ liệu (Vectorized)
    # Scale Amount và Time giống logic model
    X = df[[amt_col, time_col]].values / [100.0, 1000.0]
    V = df[[f'V{i}' for i in range(1, 29)]].values
    input_array = np.hstack([X, V])
    
    # 2. AI Dự đoán
    probs = model.predict_proba(input_array)[:, 1]
    fraud_indices = np.where(probs > 0.5)[0]
    fraud_count = len(fraud_indices)
    
    # 3. Lưu vào DB hàng loạt nếu có gian lận
    if fraud_count > 0:
        conn = None
        try:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                # Tối ưu: Dùng list comprehension thay vì lặp iloc (iloc bên trong loop rất chậm)
                # Lấy các giá trị cần thiết ra mảng trước
                selected_rows = df.iloc[fraud_indices]
                amounts = selected_rows[amt_col].values
                times = selected_rows[time_col].values
                fraud_probs = probs[fraud_indices]
                
                insert_data = [
                    (float(amounts[i]), float(times[i]), float(fraud_probs[i]), source)
                    for i in range(len(fraud_indices))
                ]
                
                execute_values(cur, 
                    "INSERT INTO system_fraud_logs (amount, time_val, fraud_probability, source) VALUES %s", 
                    insert_data)
                conn.commit()
                cur.close()
        except: pass
        finally:
            if conn: release_db_connection(conn)
            
    return fraud_count

@st.cache_data
def load_csv_data(file):
    return pd.read_csv(file)

# --- 3. UI ---

# Header Cố định
st.markdown("""
<div class="custom-header">
    <div class="header-branding">SafeGuard Banking</div>
    <div class="header-right">
        <div class="header-icons">
            <div class="icon-wrapper" style="margin-right: 15px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
                <div class="icon-dot"></div>
            </div>
            <div class="icon-wrapper">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
            </div>
        </div>
        <div class="user-block">
            <div class="user-info">
                <div class="user-name">Administrator</div>
                <div class="user-role">Quản trị viên</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_sep, col_right = st.columns([1.2, 0.1, 2.7])

# CỘT TRÁI: LIVE MONITORING
with col_left:
    @st.fragment(run_every=30)
    def live_monitoring_panel():
        st.markdown('<div class="live-monitor-title"><span class="live-dot"></span> Giám sát Realtime</div>', unsafe_allow_html=True)
        alerts = get_api_alerts()
        
        if not alerts:
            st.info("Chưa có cảnh báo nào từ API...")
        else:
            for amt, prob, ts, src in alerts:
                time_str = ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)
                st.markdown(f"""
                <div class="alert-card">
                    <div class="alert-source">{src}</div>
                    <div style="font-size:0.85rem; font-weight:600;">Giao dịch gian lận!</div>
                    <div class="alert-meta">Số tiền: <b>€{amt:,.2f}</b> • Prob: <b>{prob if isinstance(prob, str) else f"{prob:.1%}"}</b><br>{time_str}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("---")

    live_monitoring_panel()

# CỘT GIỮA: VẠCH PHÂN CÁCH
with col_sep:
    st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

# CỘT PHẢI: ANALYSIS CENTER
with col_right:
    @st.fragment()
    def analysis_center_cloud():
        st.markdown('<div class="section-header" style="justify-content: center;">Phân tích Giao dịch</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Kiểm Tra Thủ Công", "Tải Lên File"])
        
        with tab1:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            c_base1, c_base2 = st.columns(2)
            with c_base1: st.number_input("Số tiền", value=100.0, step=None, key="amt_cloud")
            with c_base2: st.number_input("Thời gian", value=1000.0, step=None, key="time_cloud")
            
            selected_vs = st.multiselect(
                "Chọn thêm đặc trưng để nhập dữ liệu:",
                options=[f"V{i}" for i in range(1, 29)],
                default=["V17", "V14", "V16", "V12"],
                key="v_multi_cloud"
            )
            
            if selected_vs:
                v_cols = st.columns(4)
                for i, v_name in enumerate(selected_vs):
                    with v_cols[i % 4]:
                        st.markdown(f"""
                            <div class="v-feature-row">
                                <span class="feature-name">{v_name}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        if st.button("✕", key=f"del_{v_name}_cloud", type="secondary"):
                            st.session_state.v_multi_cloud.remove(v_name)
                            st.rerun()
                        st.number_input(v_name, value=0.0, step=None, label_visibility="collapsed", key=f"val_{v_name}_cloud")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Bắt đầu phân tích", type="primary", key="btn_cloud"):
                with st.spinner("Đang phân tích..."):
                    v_feats = [0.0]*28
                    for v_name in selected_vs:
                        v_idx = int(v_name[1:])
                        v_feats[v_idx-1] = st.session_state[f"val_{v_name}_cloud"]
                    res = process_prediction(st.session_state.amt_cloud, st.session_state.time_cloud, v_feats, source="HỆ THỐNG (Manual)")
                    if res:
                        if "BLOCK" in res['decision']:
                            st.error(f"Kết quả: GIAN LẬN ({res['prob']} gian lận)")
                        else:
                            st.success(f"Kết quả: HỢP LỆ ({res['prob']} gian lận)")

        with tab2:
            up = st.file_uploader("Tải lên file giao dịch (.csv)", type="csv", key="file_cloud", label_visibility="collapsed")
            if up:
                df = load_csv_data(up)
                st.dataframe(df.head(), use_container_width=True)
                st.markdown('<div class="center-btn">', unsafe_allow_html=True)
                if st.button("Quét toàn bộ tập tin", type="primary", key="scan_cloud"):
                    with st.spinner("Đang phân tích hàng loạt..."):
                        # Nhận diện cột
                        amt_col = 'Amount' if 'Amount' in df.columns else 'scaled_amount'
                        time_col = 'Time' if 'Time' in df.columns else 'scaled_time'
                        
                        batch_size = 100_000
                        fraud_total = 0
                        total_rows = len(df)
                        
                        for start_idx in range(0, total_rows, batch_size):
                            end_idx = min(start_idx + batch_size, total_rows)
                            batch_df = df.iloc[start_idx:end_idx]
                            
                            fraud_total += process_bulk_cloud(batch_df, amt_col, time_col)
                        
                        st.success(f"Hoàn tất! Đã xử lý {total_rows:,} giao dịch. Phát hiện {fraud_total} vụ gian lận.")
                st.markdown('</div>', unsafe_allow_html=True)

    analysis_center_cloud()
