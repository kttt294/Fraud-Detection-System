import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import time
import io

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
                        log_id = alert.get('id')
                        confirmed = alert.get('confirmed')
                        ts_str = alert.get('created_at', '')
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            time_display = dt.strftime("%H:%M:%S")
                        except:
                            time_display = ts_str

                        # Badge trạng thái
                        if confirmed is True:
                            badge = '<span style="background:#dcfce7;color:#16a34a;font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:4px;">✓ ĐÃ XÁC NHẬN</span>'
                        else:
                            badge = '<span style="background:#fef9c3;color:#ca8a04;font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:4px;">⏳ CHờ XÁC NHẬN</span>'

                        # Container bao quanh mỗi Alert
                        with st.container(border=True):
                            st.markdown(f"""
                                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                                    <span style="font-size: 0.65rem; font-weight: 700; color: #64748b; text-transform: uppercase;">{alert.get('source', 'API')}</span>
                                    {badge}
                                </div>
                                <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b; margin-bottom: 2px;">Giao dịch gian lận!</div>
                                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 8px;">
                                    Số tiền: <b>€{alert.get('amount', 0):,.2f}</b> • Prob: <b>{alert.get('fraud_probability', 0):.1%}</b><br>
                                    🕒 {time_display}
                                </div>
                            """, unsafe_allow_html=True)

                            if confirmed is not True:
                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button("✅ Xác nhận", key=f"confirm_{log_id}", use_container_width=True):
                                        try:
                                            requests.put(f"{API_BASE_URL}/confirm-fraud/{log_id}", json={"is_fraud": True}, timeout=3)
                                            st.rerun()
                                        except:
                                            st.error("Lỗi kết nối!")
                                with c2:
                                    if st.button("❌ Báo giả", key=f"reject_{log_id}", use_container_width=True, type="secondary"):
                                        try:
                                            requests.put(f"{API_BASE_URL}/confirm-fraud/{log_id}", json={"is_fraud": False}, timeout=3)
                                            st.rerun()
                                        except:
                                            st.error("Lỗi kết nối!")
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
            c_base1, b_base2 = st.columns([1, 1.2])
            with c_base1:
                st.markdown('<p style="margin:0 0 4px 0;font-weight:600;font-size:0.9rem;color:#1e293b">Số tiền (Amount)</p>', unsafe_allow_html=True)
                st.number_input("Số tiền", value=100.0, step=None, label_visibility="collapsed", key="amt_front")
            with b_base2:
                st.markdown('<p style="margin:0 0 4px 0;font-weight:600;font-size:0.9rem;color:#1e293b">Giờ giao dịch (H:M:S)</p>', unsafe_allow_html=True)
                st.markdown('<div class="time-input-container">', unsafe_allow_html=True)
                tc1, tc2, tc3, _ = st.columns([1, 1, 1, 5])
                now = datetime.now()
                with tc1: h = st.number_input("H", min_value=0, max_value=23, value=now.hour, key="h_front", label_visibility="collapsed")
                with tc2: m = st.number_input("M", min_value=0, max_value=59, value=now.minute, key="m_front", label_visibility="collapsed")
                with tc3: s = st.number_input("S", min_value=0, max_value=59, value=now.second, key="s_front", label_visibility="collapsed")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div data-testid="v_feature_container">', unsafe_allow_html=True)
            selected_vs = st.multiselect(
                "Chọn thêm đặc trưng để nhập dữ liệu:",
                options=[f"V{i}" for i in range(1, 29)],
                default=["V17", "V14", "V16", "V12"],
                key="v_multi_front"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
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
                    tx_time = f"{st.session_state.h_front:02d}:{st.session_state.m_front:02d}:{st.session_state.s_front:02d}"
                    payload = {
                        "amount": st.session_state.amt_front,
                        "transaction_time": tx_time,
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
                        all_detected_frauds = []
                        total = len(df)
                        v_cols = [f'V{i}' for i in range(1, 28 + 1)]
                        
                        for start_idx in range(0, total, batch_size):
                            end_idx = min(start_idx + batch_size, total)
                            batch_df = df.iloc[start_idx:end_idx]
                            
                            transactions = []
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
                                    batch_frauds = res.get("frauds", [])
                                    if batch_frauds:
                                        all_detected_frauds.extend(batch_frauds)
                            except Exception as e:
                                st.error(f"Lỗi khi gửi lô {start_idx}-{end_idx}: {e}")
                        
                        if fraud_count > 0:
                            st.session_state.fraud_df_front = pd.DataFrame(all_detected_frauds) if all_detected_frauds else pd.DataFrame()
                            st.success(f"Hoàn tất! Đã xử lý {total:,} giao dịch. Phát hiện {fraud_count} vụ gian lận.")
                            if not all_detected_frauds:
                                st.warning("⚠️ Đã phát hiện gian lận nhưng không thể lấy danh sách chi tiết. Vui lòng khởi động lại Backend API.")
                        else:
                            st.session_state.fraud_df_front = pd.DataFrame()
                            st.info(f"Hoàn tất! Không phát hiện gian lận trong {total:,} giao dịch.")

                # Hiển thị nút tải xuống nếu có kết quả
                if 'fraud_df_front' in st.session_state and not st.session_state.fraud_df_front.empty:
                    st.write("---")
                    st.markdown('<p style="font-weight:600; color:#1e293b;">Xuất kết quả gian lận:</p>', unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        fmt = st.selectbox("Định dạng file:", ["CSV", "Excel", "JSON"], key="fmt_front")
                    
                    with c2:
                        st.write("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Spacer
                        file_data = None
                        mime_type = ""
                        file_ext = fmt.lower()
                        
                        if fmt == "CSV":
                            file_data = st.session_state.fraud_df_front.to_csv(index=False).encode('utf-8')
                            mime_type = "text/csv"
                        elif fmt == "Excel":
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                st.session_state.fraud_df_front.to_excel(writer, index=False)
                            file_data = output.getvalue()
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            file_ext = "xlsx"
                        elif fmt == "JSON":
                            file_data = st.session_state.fraud_df_front.to_json(orient='records', indent=4).encode('utf-8')
                            mime_type = "application/json"

                        st.download_button(
                            label=f"Tải file {fmt}",
                            data=file_data,
                            file_name=f"local_detected_frauds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type,
                            use_container_width=True
                        )

    analysis_center_frontend()
