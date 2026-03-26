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

# Thêm Inline CSS để cưỡng ép giao diện nút bấm giống bản Cloud
st.markdown("""
<style>
/* Nút phụ - Xác nhận (Gray, Mini) */
.stButton button[kind="secondary"] {
    width: 100% !important; 
    max-width: 90px !important;
    margin-left: auto !important;
    display: block !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important; /* Đồng bộ kích thước */
    height: 24px !important; /* Tăng nhẹ để cân xứng với chữ 0.7rem */
    line-height: 1 !important;
    padding: 0 !important;
    background-color: #f1f5f9 !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.2s ease !important;
}

/* Nút chính - Phân tích (Blue, To rõ ràng như cũ) */
.stButton button[kind="primary"] {
    width: 100% !important; 
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    height: 38px !important;
    background-color: #38bdf8 !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.stButton button:hover {
    filter: brightness(0.95);
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.5rem 0.8rem !important;
}
.time-label {
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 0 0 2px 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

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
                            badge = '<span style="background:#dcfce7;color:#16a34a;font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:4px;">ĐÃ XÁC NHẬN</span>'
                        else:
                            badge = '<span style="background:#fef9c3;color:#ca8a04;font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:4px;">⏳ CHờ XÁC NHẬN</span>'

                        # Container bao quanh mỗi Alert
                        with st.container(border=True):
                            # Header: Source + Nút Xác nhận
                            h1, h2 = st.columns([2.2, 1.4])
                            with h1:
                                st.markdown(f'<span style="font-size: 0.65rem; font-weight: 700; color: #64748b; text-transform: uppercase;">{alert.get("source", "API")}</span>', unsafe_allow_html=True)
                            with h2:
                                if confirmed is True:
                                    st.markdown('<span style="background:#dcfce7;color:#16a34a;font-size:0.7rem;font-weight:700;padding:2px 4px;border-radius:4px;display:block;text-align:center;">ĐÃ XÁC NHẬN</span>', unsafe_allow_html=True)
                                else:
                                    if st.button("Xác nhận", key=f"conf_btn_{log_id}", use_container_width=True):
                                        try:
                                            requests.put(f"{API_BASE_URL}/confirm-fraud/{log_id}", json={"is_fraud": True}, timeout=3)
                                            st.rerun()
                                        except:
                                            st.error("Lỗi!")
                            
                            # Body (Compact)
                            st.markdown(f"""
                                <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b; margin: 2px 0 0 0;">Giao dịch gian lận!</div>
                                <div style="font-size: 0.72rem; color: #64748b; margin: 0;">
                                    Số tiền: <b>€{alert.get('amount', 0):,.2f}</b> • Prob: <b>{alert.get('fraud_probability', 0):.1%}</b> | 🕒 {time_display}
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
            st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
            # Chia làm 4 cột trên cùng 1 hàng
            now = datetime.now()
            ca, ch, cm, cs = st.columns([1.8, 1, 1, 1])
            
            with ca:
                st.markdown('<p style="margin:0 0 4px 0;font-weight:600;font-size:0.85rem;color:#1e293b">Số tiền (Amount)</p>', unsafe_allow_html=True)
                st.number_input("Số tiền", value=100.0, step=None, label_visibility="collapsed", key="amt_front")
            with ch:
                st.markdown('<p class="time-label">Giờ (H)</p>', unsafe_allow_html=True)
                h = st.number_input("H", min_value=0, max_value=23, value=now.hour, key="h_front", label_visibility="collapsed")
            with cm:
                st.markdown('<p class="time-label">Phút (M)</p>', unsafe_allow_html=True)
                m = st.number_input("M", min_value=0, max_value=59, value=now.minute, key="m_front", label_visibility="collapsed")
            with cs:
                st.markdown('<p class="time-label">Giây (S)</p>', unsafe_allow_html=True)
                s = st.number_input("S", min_value=0, max_value=59, value=now.second, key="s_front", label_visibility="collapsed")
            
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
                try:
                    with st.spinner("Đang xử lý dữ liệu..."):
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
                        
                        # Thêm Timeout 3s để tránh bị treo nút bấm
                        res_all = requests.post(f"{API_BASE_URL}/verify", json=payload, timeout=3).json()
                    
                    # Hiển thị kết quả ngoài spinner để đảm bảo UX
                    if res_all.get("decision") == "BLOCK":
                        st.error(f"CẢNH BÁO: GIAN LẬN ({res_all.get('probability')} xác suất)")
                    else:
                        st.success(f"✅ HỢP LỆ ({res_all.get('probability')} xác suất gian lận)")
                except requests.exceptions.Timeout:
                    st.error("⏳ Lỗi: Backend phản hồi quá chậm (Timeout 3s).")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối: {str(e)}")
        
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
                                st.warning("Đã phát hiện gian lận nhưng không thể lấy danh sách chi tiết. Vui lòng khởi động lại Backend API.")
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
                            use_container_width=True,
                            type="primary"
                        )

    analysis_center_frontend()
