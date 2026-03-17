import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import time

# --- 1. CONFIG & SETUP ---
# --- 1. CONFIG & SETUP ---
st.set_page_config(page_title="SafeGuard Banking | Monitoring", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(__file__), 'style.css')
if os.path.exists(css_path):
    with open(css_path, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

API_BASE_URL = "http://localhost:8000"

@st.cache_data
def load_csv_data(file):
    """Ghi nhớ dữ liệu file vào RAM để không phải đọc lại mỗi lần nhấn nút."""
    return pd.read_csv(file)

# --- 2. UI LAYOUT ---

# Header
st.markdown(f"""
<div class="custom-header">
    <div class="header-branding">SafeGuard Banking | Monitoring Dashboard</div>
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

# Layout chính 3 cột (Live | Divider | Analysis)
col_left, col_sep, col_right = st.columns([1.2, 0.1, 2.7])

with col_left:
    @st.fragment(run_every=30)
    def live_monitoring_frontend():
        st.markdown('<div class="live-monitor-title"><span class="live-dot"></span> Giám sát Realtime</div>', unsafe_allow_html=True)
        # Lấy dữ liệu từ Backend
        try:
            response = requests.get(f"{API_BASE_URL}/alerts?limit=8", timeout=2)
            if response.status_code == 200:
                api_alerts = response.json().get('data', [])
                
                if not api_alerts:
                    st.info("Chưa có cảnh báo nào từ API...")
                else:
                    for alert in api_alerts:
                        # Định dạng thời gian từ chuỗi ISO
                        ts_str = alert.get('created_at', '')
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            time_display = dt.strftime("%H:%M:%S")
                        except:
                            time_display = ts_str

                        st.markdown(f"""
                        <div class="alert-card">
                            <div class="alert-source">{alert.get('source', 'API')}</div>
                            <div style="font-size:0.85rem; font-weight:600;">Giao dịch gian lận!</div>
                            <div class="alert-meta">
                                Số tiền: <b>€{alert.get('amount', 0):,.2f}</b> • Prob: <b>{alert.get('fraud_probability', 0):.1%}</b>
                                <br>{time_display}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        except:
            st.warning("Đang kết nối tới Live API...")
        st.write("---")

    live_monitoring_frontend()

# CỘT GIỮA: VẠCH PHÂN CÁCH
with col_sep:
    st.markdown('<div class="vertical-divider"></div>', unsafe_allow_html=True)

with col_right:
    @st.fragment()
    def analysis_center_frontend():
        st.markdown('<div class="section-header" style="justify-content: center;">Phân tích Giao dịch</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Kiểm Tra Thủ Công", "Tải Lên File"])
        
        with tab1:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            c_base1, c_base2 = st.columns(2)
            with c_base1: st.number_input("Số tiền", value=100.0, step=None, key="amt_front")
            with c_base2: st.number_input("Thời gian", value=1000.0, step=None, key="time_front")
            
            selected_vs = st.multiselect(
                "Chọn thêm đặc trưng để nhập dữ liệu:",
                options=[f"V{i}" for i in range(1, 29)],
                default=["V17", "V14", "V16", "V12"],
                key="v_multi_front"
            )
            
            if selected_vs:
                v_cols = st.columns(4)
                for i, v_name in enumerate(selected_vs):
                    with v_cols[i % 4]:
                        # Sử dụng class .v-feature-row từ style.css
                        st.markdown(f"""
                            <div class="v-feature-row">
                                <span class="feature-name">{v_name}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("✕", key=f"del_{v_name}_front", type="secondary"):
                            st.session_state.v_multi_front.remove(v_name)
                            st.rerun()
                        
                        st.number_input(v_name, value=0.0, step=None, label_visibility="collapsed", key=f"val_{v_name}_front")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Bắt đầu phân tích", type="primary", key="btn_front"):
                with st.spinner("Đang phân tích..."):
                    payload = {
                        "amount": st.session_state.amt_front,
                        "time_val": st.session_state.time_front,
                        "v_features": [0.0]*28,
                        "source": "Phân tích Thủ công"
                    }
                    # Thu thập tất cả các V-features đã chọn
                    for v_name in selected_vs:
                        v_idx = int(v_name[1:])
                        payload["v_features"][v_idx-1] = st.session_state[f"val_{v_name}_front"]
                    
                    try:
                        res = requests.post(f"{API_BASE_URL}/verify", json=payload, timeout=5).json()
                        if res.get("decision") == "BLOCK":
                            st.error(f"Kết quả: GIAN LẬN ({res.get('probability')} gian lận)")
                        else:
                            st.success(f"Kết quả: HỢP LỆ ({res.get('probability')} gian lận)")
                    except:
                        st.error("Không thể kết nối đến Backend API!")
        
        with tab2:
            up = st.file_uploader("Chọn file CSV", type="csv", key="file_front", label_visibility="collapsed")
            if up:
                df = load_csv_data(up)
                st.dataframe(df.head(), use_container_width=True)
                if st.button("Quét toàn bộ tập tin", key="scan_front", type="primary"):
                    with st.spinner("Đang phân tích..."):
                        # Tự động nhận diện cột
                        amt_col = 'Amount' if 'Amount' in df.columns else 'scaled_amount'
                        time_col = 'Time' if 'Time' in df.columns else 'scaled_time'
                        
                        batch_size = 100_000
                        fraud_count = 0
                        total = len(df)
                        v_cols = [f'V{i}' for i in range(1, 28 + 1)]
                        
                        for start_idx in range(0, total, batch_size):
                            end_idx = min(start_idx + batch_size, total)
                            batch_df = df.iloc[start_idx:end_idx]
                            
                            # Tối ưu: Chuyển sang dict trực tiếp từ DataFrame thay vì dùng iterrows
                            transactions = []
                            # chuẩn bị dữ liệu mảng V
                            v_array = batch_df[v_cols].values.tolist()
                            amounts = batch_df[amt_col].values.astype(float)
                            times = batch_df[time_col].values.astype(float)
                            
                            for i in range(len(batch_df)):
                                transactions.append({
                                    "amount": amounts[i],
                                    "time_val": times[i],
                                    "v_features": v_array[i],
                                    "source": "Quét tập tin"
                                })
                            
                            try:
                                payload = {"transactions": transactions}
                                res = requests.post(f"{API_BASE_URL}/verify-bulk", json=payload, timeout=30).json()
                                if res.get("status") == "success":
                                    fraud_count += res.get("fraud_detected", 0)
                            except Exception as e:
                                st.error(f"Lỗi khi gửi lô {start_idx}-{end_idx}: {e}")
                        
                        st.success(f"Hoàn tất! Đã xử lý {total:,} giao dịch thông qua API. Phát hiện {fraud_count} vụ gian lận.")

    analysis_center_frontend()
