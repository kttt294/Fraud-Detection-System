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
import json
import requests
from datetime import datetime
import io

# --- 1. CONFIG & SETUP ---
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
st.set_page_config(page_title="SafeGuard Banking", layout="wide")

# NHÚNG TOÀN BỘ CSS
CSS_EMBEDDED = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: white !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    padding: 60px 40px 20px 40px !important;
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
    max-width: 100% !important;
    margin-bottom: 10px !important;
}

.stButton:has(> button[data-testid="stBaseButton-primary"]) {
    display: flex;
    justify-content: center;
}

.stButton > button[data-testid="stBaseButton-primary"] {
    width: 100% !important; 
    background-color: #38bdf8 !important;
    color: white !important; border-radius: 6px !important;
    border: none !important; font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}

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

.center-btn {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* FIX: Nút Xác nhận / Báo giả trong alert cards — width auto, không bóp méo */
.stButton > button[data-testid="stBaseButton-secondary"] {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: none !important;
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 4px 10px !important;
    min-width: unset !important;
    width: auto !important;
    height: 32px !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.4 !important;
}

.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #fee2e2 !important;
    border-color: #fca5a5 !important;
    color: #ef4444 !important;
}

span[data-baseweb="tag"], button[aria-label="Clear all"] {
    display: none !important;
}

div[data-testid="stNumberInput"] button {
    display: none !important;
}

/* FIX: Label phụ cho ô giờ/phút/giây */
.time-label {
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 0 0 2px 0;
    font-weight: 500;
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
def load_scaler():
    scaler_path = 'modeling/scaler.pkl'
    scaler_url = "https://github.com/kttt294/SafeGuard-Fraud-Detection-System/releases/download/v1.0.0/scaler.pkl"
    
    try:
        if not os.path.exists(scaler_path):
            os.makedirs('modeling', exist_ok=True)
            with st.spinner("Đang tải Scaler từ GitHub..."):
                resp = requests.get(scaler_url, stream=True)
                if resp.status_code == 200:
                    with open(scaler_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    st.error(f"Lỗi tải scaler từ GitHub: {resp.status_code}")
                    return None
                    
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi khởi tạo scaler: {e}")
    return None

scaler = load_scaler()

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
        
        if os.path.exists(ca_path) and ssl_mode == "require":
            params["sslrootcert"] = ca_path
        elif ssl_mode == "require":
            params["sslmode"] = "allow"

        return pool.SimpleConnectionPool(1, 20, **params)
    except: return None

@st.cache_resource
def init_db_cloud():
    conn = None
    try:
        conn = get_db_pool().getconn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                v_features JSONB,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'API (User App)',
                confirmed BOOLEAN
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_fraud_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                amount FLOAT,
                time_val FLOAT,
                v_features JSONB,
                fraud_probability FLOAT,
                source TEXT DEFAULT 'Dashboard System',
                confirmed BOOLEAN
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
            cur.execute("SELECT id, amount, fraud_probability, created_at, source, confirmed FROM api_fraud_logs ORDER BY created_at DESC LIMIT 8")
            rows = cur.fetchall()
            cur.close()
            return rows
    except: pass
    finally:
        if conn: release_db_connection(conn)
    return []

def confirm_fraud_db(log_id: int, is_fraud: bool, table: str = "api_fraud_logs"):
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE {table} SET confirmed = %s WHERE id = %s", (is_fraud, log_id))
            conn.commit()
            cur.close()
    except: pass
    finally:
        if conn: release_db_connection(conn)

def process_prediction(amount, transaction_time, v_features, source="HỆ THỐNG (Manual)"):
    if model:
        conn = None
        try:
            if transaction_time:
                try:
                    parts = transaction_time.strip().split(":")
                    h, m, s = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
                    time_val = float(h * 3600 + m * 60 + s)
                except Exception:
                    now = datetime.now()
                    time_val = float(now.hour * 3600 + now.minute * 60 + now.second)
            else:
                now = datetime.now()
                time_val = float(now.hour * 3600 + now.minute * 60 + now.second)

            if scaler:
                n_feats = getattr(scaler, 'n_features_in_', 1)
                if n_feats == 2:
                    input_scaled = scaler.transform([[amount, time_val]])
                    scaled_amt = float(input_scaled[0][0])
                    scaled_time = float(input_scaled[0][1])
                else:
                    scaled_amt = float(scaler.transform([[amount]])[0][0])
                    scaled_time = float(scaler.transform([[time_val]])[0][0])
            else:
                scaled_amt = amount / 100
                scaled_time = time_val / 86400

            input_data = [scaled_amt, scaled_time] + v_features
            features_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
            prob = model.predict_proba(features_df)[0][1]
            decision = "BLOCK" if prob > 0.5 else "APPROVE"

            if decision == "BLOCK":
                conn = get_db_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO system_fraud_logs (amount, time_val, v_features, fraud_probability, source) VALUES (%s, %s, %s, %s, %s)",
                        (amount, time_val, json.dumps(v_features), prob, source)
                    )
                    conn.commit()
                    cur.close()
            return {"decision": decision, "prob": f"{prob:.2%}"}
        except: pass
        finally:
            if conn: release_db_connection(conn)
    return None

def process_bulk_cloud(df, amt_col, time_col, source="HỆ THỐNG (Bulk)"):
    if not model: return 0
    if scaler and amt_col in ('Amount', 'Time'):
        n_feats = getattr(scaler, 'n_features_in_', 1)
        if n_feats == 2:
            X_raw = df[[amt_col, time_col]].values
            X = scaler.transform(X_raw)
        else:
            amt_scaled = scaler.transform(df[[amt_col]].values).flatten()
            time_scaled = scaler.transform(df[[time_col]].values).flatten()
            X = np.column_stack([amt_scaled, time_scaled])
    else:
        X = df[[amt_col, time_col]].values
    V = df[[f'V{i}' for i in range(1, 29)]].values
    input_array = np.hstack([X, V])
    
    probs = model.predict_proba(input_array)[:, 1]
    fraud_indices = np.where(probs > 0.5)[0]
    fraud_count = len(fraud_indices)
    
    fraud_df_batch = df.iloc[fraud_indices].copy()
    if fraud_count > 0:
        fraud_df_batch['fraud_probability'] = probs[fraud_indices]
    
    if fraud_count > 0:
        conn = None
        try:
            conn = get_db_connection()
            if conn:
                cur = conn.cursor()
                selected_rows = df.iloc[fraud_indices]
                amounts = selected_rows[amt_col].values
                times = selected_rows[time_col].values
                fraud_probs = probs[fraud_indices]
                v_feats_list = selected_rows[[f'V{i}' for i in range(1, 29)]].values.tolist()
                
                insert_data = [
                    (float(amounts[i]), float(times[i]), json.dumps(v_feats_list[i]), float(fraud_probs[i]), source)
                    for i in range(len(fraud_indices))
                ]
                
                execute_values(cur, 
                    "INSERT INTO system_fraud_logs (amount, time_val, v_features, fraud_probability, source) VALUES %s", 
                    insert_data)
                conn.commit()
                cur.close()
        except: pass
        finally:
            if conn: release_db_connection(conn)
            
    return fraud_df_batch

@st.cache_data
def load_csv_data(file):
    return pd.read_csv(file)

# --- 3. UI ---

st.markdown("""
<div class="custom-header">
    <div class="header-branding">SafeGuard - Credit Fraud Detection</div>
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

with col_left:
    @st.fragment(run_every=30)
    def live_monitoring_panel():
        st.markdown('<div class="live-monitor-title"><span class="live-dot"></span> Giám sát Realtime</div>', unsafe_allow_html=True)
        alerts = get_api_alerts()

        if not alerts:
            st.info("Chưa có cảnh báo nào từ API...")
        else:
            for row in alerts:
                log_id, amt, prob, ts, src, confirmed = row
                time_str = ts.strftime("%H:%M:%S") if isinstance(ts, datetime) else str(ts)
                prob_str = prob if isinstance(prob, str) else f"{prob:.1%}"

                # Hiển thị Card cảnh báo
                with st.container(border=True):
                    # Header: Source + Nút Xác nhận ở góc phải
                    h1, h2 = st.columns([3, 1])
                    with h1:
                        st.markdown(f'<span style="font-size: 0.65rem; font-weight: 700; color: #64748b; text-transform: uppercase;">{src}</span>', unsafe_allow_html=True)
                    with h2:
                        if confirmed is True:
                            st.markdown('<span style="background:#dcfce7;color:#16a34a;font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:4px;display:block;text-align:center;">✓ ĐÃ XÁC NHẬN</span>', unsafe_allow_html=True)
                        else:
                            if st.button("Xác nhận", key=f"conf_btn_{log_id}", use_container_width=True):
                                confirm_fraud_db(log_id, True)
                                st.rerun()

                    # Body
                    st.markdown(f"""
                        <div style="font-size: 0.9rem; font-weight: 600; color: #1e293b; margin-top: 4px;">Giao dịch gian lận!</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 2px;">
                            Số tiền: <b>€{amt:,.2f}</b> • Prob: <b>{prob_str}</b><br>
                            🕒 {time_str}
                        </div>
                    """, unsafe_allow_html=True)

        st.write("---")

    live_monitoring_panel()

with col_sep:
    st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

with col_right:
    @st.fragment()
    def analysis_center_cloud():
        st.markdown('<div class="section-header" style="justify-content: center;">Phân tích Giao dịch</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Kiểm Tra Thủ Công", "Tải Lên File"])
        
        with tab1:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            c_base1, b_base2 = st.columns([1, 1.2])
            with c_base1:
                st.markdown('<p style="margin:0 0 4px 0;font-weight:600;font-size:0.9rem;color:#1e293b">Số tiền (Amount)</p>', unsafe_allow_html=True)
                st.number_input("Số tiền", value=100.0, step=None, label_visibility="collapsed", key="amt_cloud")

            # FIX: Bỏ cột thứ 4 trống, dùng st.columns(3) chia đều — tránh 3 ô bị bóp
            with b_base2:
                st.markdown('<p style="margin:0 0 4px 0;font-weight:600;font-size:0.9rem;color:#1e293b">Giờ giao dịch (H:M:S)</p>', unsafe_allow_html=True)
                now = datetime.now()
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.markdown('<p class="time-label">Giờ</p>', unsafe_allow_html=True)
                    h = st.number_input("H", min_value=0, max_value=23, value=now.hour, key="h_cloud", label_visibility="collapsed")
                with tc2:
                    st.markdown('<p class="time-label">Phút</p>', unsafe_allow_html=True)
                    m = st.number_input("M", min_value=0, max_value=59, value=now.minute, key="m_cloud", label_visibility="collapsed")
                with tc3:
                    st.markdown('<p class="time-label">Giây</p>', unsafe_allow_html=True)
                    s = st.number_input("S", min_value=0, max_value=59, value=now.second, key="s_cloud", label_visibility="collapsed")
            
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
                        st.markdown(f'<p style="margin:0 0 4px 0;font-weight:600;font-size:0.9rem;color:#1e293b">{v_name}</p>', unsafe_allow_html=True)
                        st.number_input(v_name, value=0.0, step=None, label_visibility="collapsed", key=f"val_{v_name}_cloud")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Bắt đầu phân tích", type="primary", key="btn_cloud"):
                with st.spinner("Đang phân tích..."):
                    v_feats = [0.0]*28
                    for v_name in selected_vs:
                        v_idx = int(v_name[1:])
                        v_feats[v_idx-1] = st.session_state[f"val_{v_name}_cloud"]
                    tx_time = f"{st.session_state.h_cloud:02d}:{st.session_state.m_cloud:02d}:{st.session_state.s_cloud:02d}"
                    res = process_prediction(st.session_state.amt_cloud, tx_time, v_feats, source="HỆ THỐNG (Manual)")
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
                        amt_col = 'Amount' if 'Amount' in df.columns else 'scaled_amount'
                        time_col = 'Time' if 'Time' in df.columns else 'scaled_time'
                        
                        batch_size = 100_000
                        all_frauds = []
                        total_rows = len(df)
                        
                        for start_idx in range(0, total_rows, batch_size):
                            end_idx = min(start_idx + batch_size, total_rows)
                            batch_df = df.iloc[start_idx:end_idx]
                            fraud_batch = process_bulk_cloud(batch_df, amt_col, time_col)
                            if not fraud_batch.empty:
                                all_frauds.append(fraud_batch)
                        
                        if all_frauds:
                            st.session_state.fraud_df_cloud = pd.concat(all_frauds, ignore_index=True)
                            st.success(f"Hoàn tất! Đã xử lý {total_rows:,} giao dịch. Phát hiện {len(st.session_state.fraud_df_cloud)} vụ gian lận.")
                        else:
                            st.session_state.fraud_df_cloud = pd.DataFrame()
                            st.info(f"Hoàn tất! Không phát hiện gian lận trong {total_rows:,} giao dịch.")

                if 'fraud_df_cloud' in st.session_state and not st.session_state.fraud_df_cloud.empty:
                    st.write("---")
                    st.markdown('<p style="font-weight:600; color:#1e293b;">Xuất kết quả gian lận:</p>', unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fmt = st.selectbox("Định dạng file:", ["CSV", "Excel", "JSON"], key="fmt_cloud")
                    with c2:
                        st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        file_data = None
                        mime_type = ""
                        file_ext = fmt.lower()
                        if fmt == "CSV":
                            file_data = st.session_state.fraud_df_cloud.to_csv(index=False).encode('utf-8')
                            mime_type = "text/csv"
                        elif fmt == "Excel":
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                st.session_state.fraud_df_cloud.to_excel(writer, index=False)
                            file_data = output.getvalue()
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            file_ext = "xlsx"
                        elif fmt == "JSON":
                            file_data = st.session_state.fraud_df_cloud.to_json(orient='records', indent=4).encode('utf-8')
                            mime_type = "application/json"
                        st.download_button(
                            label=f"Tải file {fmt}",
                            data=file_data,
                            file_name=f"detected_frauds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type,
                            use_container_width=True
                        )
                st.markdown('</div>', unsafe_allow_html=True)

    analysis_center_cloud()