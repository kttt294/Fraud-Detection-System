<div align="center">

# SafeGuard: Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng
(Credit Card Fraud Detection System)


LINK COLAB CỦA DỰ ÁN:  *https://colab.research.google.com/drive/1EeArucOHdVqChyhmykQwrgyxkwX2VVGA*


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kttt294-fraud-detection.streamlit.app) [![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-%231E90FF.svg?style=flat)](https://xgboost.readthedocs.io/)

*Dự án Bài tập lớn Môn Học Máy - Đại học Phenikaa (Nhóm 3)*

Giảng viên hướng dẫn: **PGS.TS. Phạm Tiến Lâm**

Thành viên nhóm:

**Kiều Thị Thu Trang** (24100093) - *Trưởng nhóm*
**Trần Minh Sang** (24100012) - *Thành viên*
**Ngô Quang Thiện** (24102651) - *Thành viên*

---

## 1. Giới thiệu Dự án

Gian lận thẻ tín dụng đang gây thất thoát hàng tỷ USD mỗi năm nhưng việc phát hiện bằng các hệ thống dựa trên quy tác (Rule-based) đang tỏ ra kém hiệu quả và bỏ lọt rất nhiều hành vi gian lận tinh vi.

Dự án **SafeGuard** được xây dựng nhằm mô phỏng và giải quyết 2 bài toán khốc liệt nhất mang tính đặc thù ngành tài chính hiện nay:

- **Bảo mật Dữ liệu (Data Privacy):** Mô hình buộc phải học trên dữ liệu đã được mã hóa **PCA (V1 đến V28)** để bảo mật thông tin định danh của khách hàng.
- **Mất cân bằng cực độ (Extreme Imbalance):** Tỷ lệ giao dịch gian lận cực kỳ thưa thớt (chỉ **0.172%**). Nếu thiếu sự tinh chỉnh hệ số phạt, mô hình truyền thống sẽ bị hiện tượng ảo ảnh độ chính xác (Bias), đoán mọi thứ là hợp lệ và bỏ lọt toàn bộ gian lận.

Dự án này mang đến một luồng gió mới bằng việc thiết kế một **Kiến trúc học máy tinh vi (Self-Ensemble Focal Loss)**, kết hợp tự động hoá với Hyperparameter Tuning và tìm kiếm **Ngưỡng quyết định (Threshold Tuning)**.

---

## 2. Nguồn Dữ liệu (Dataset)

Bộ dữ liệu cung cấp thông tin giao dịch thẻ tín dụng trong 2 ngày tại Châu Âu (tháng 9/2013).

- **Tổng số giao dịch:** 284,807 transactions.
- **Hợp lệ (Class 0):** 284,315 (99.828%).
- **Gian lận (Class 1):** 492 (0.172%).
- **Trường dữ liệu:** 30 đặc trưng (`V1` - `V28` ẩn danh qua định dạng PCA, `Time` và `Amount`).

---

## 3. Kiến trúc Đột phá: Self-Ensemble Focal Loss

Sau khi đối chiếu 8 phương án kỹ thuật phức tạp khác nhau (Base Models, Oversampling SMOTE, Class Weight, Stacking, Voting…), sự thành công cốt lõi của dự án nằm ở **Kiến trúc tự kết hợp kép Self-Ensemble Focal Loss** được nhóm tự thiết kế bằng phương pháp OOP trên nền `XGBoost`.

### Vượt qua giới hạn toán học (Numerical Instability)

Điểm yếu chí mạng khi mang Loss Function "Focal Loss" vốn dĩ thiết kế cho SGD bậc 1 của Facebook AI nạp vào cơ chế đạo hàm bậc 2 (Hessian - Newton Raphson) của XGBoost chính là sự bùng nổ số học (mẫu số bằng 0). Nhóm đã áp dụng thuật toán **Heuristic-Weighted Scaling** vào đạo hàm gốc của Log-loss hòng ép mô hình XGBoost ổn định số học trong việc phạt tỷ trọng.

### Thiết kế "Tuyệt chiêu" Kép (Wide & Deep FocalXGB)

1. **Tuyến Càn Quét (Wide - Gamma thấp = 1.0):** Tập trung truy vết các giao dịch gian lận mang tính lộ liễu, bề nổi.
2. **Tuyến Khoét Sâu (Deep - Gamma cực đoan = 1.25):** Phớt lờ mọi giao dịch dễ đoán, dồn 100% tài nguyên đào bới những mẫu thẻ tín dụng ngụy trang đánh cắp tiền siêu việt.
3. Các xác suất của hai mũi được gom chéo bằng **Soft-Voting Classifier**, đảm bảo tỷ lệ hội tụ F2-Score chạm đỉnh.

---

## 4. Quy trình Tối ưu Mạch lạc (Pipeline AutoTunerCV)

Toàn bộ quá trình từ lúc tiền xử lý (Train-Test Split) tới khâu tối ưu tham số mô hình đều được đóng vào Object khép kín mang tên **`AutoTunerCV`** qua 3 pha chạy tự động để tránh *Data Leakage*:

1. **Hiệu chỉnh Phân lớp (Hyperparameter Tuning):** Sử dụng `RandomizedSearchCV` cùng `StratifiedKFold`.
2. **Hiệu chỉnh Xác Suất Thực tế (Probability Calibration):** Chạy `cross_val_predict` dạng OOF. Áp dụng `IsotonicRegression` để nắng điểm xác suất không bị tự tin thái quá.
3. **Tìm Ngưỡng tối thượng (Threshold Tuning):** Duyệt cạn dải numpy `linspace` với 200 điểm cắt, dừng lại ở mức ngưỡng đảm bảo **F2-Score** cao nhất để triệt phá rủi ro dòng tiền do báo động giả.

> Lựa chọn cuối cùng đã chốt đỉnh Threshold tự nhiên ấn định lên mức **0.4020**.

---

## 5. Kết quả Ấn tượng (Tập Hold-out Test)

Qua quá trình huấn luyện và kiểm định chéo, mô hình ưu tú nhất ghi nhận các chỉ số hiệu suất:

| Chỉ số            | Giá trị Đạt được | Ý nghĩa nghiệp vụ trong Ngân hàng                                          |
| ------------------- | :---------------------: | -------------------------------------------------------------------------------- |
| **AUPRC**     |    **0.8726**    | Đánh giá tổng thể xuất sắc trên dữ liệu mất cân bằng                |
| **Recall**    |    **84.69%**    | Tóm gọn gần**85%** tổng lượng giao dịch mất cắp                   |
| **Precision** |    **92.22%**    | Độ chính xác đạt hơn**92%**, tỷ lệ "khoá nhầm" thẻ siêu thấp |
| **F2-Score**  |    **0.8610**    | Tối ưu bảo vệ dòng tiền (Nặng Recall hơn Precision)                      |

*(Tham khảo báo cáo chi tiết cho đối sánh với Baseline & SMOTE)*

---

## 6. Web Application: SafeGuard

Đóng gói kiến trúc Machine Learning tinh thể hoá `.pkl` và thiết kế UI chuyên biệt qua Streamlit thành cổng hỗ trợ chuyên viên phân tích tài chính.

🔗 **Link Demo Trực tiếp:** [SafeGuard Dashboard](https://kttt294-fraud-detection.streamlit.app)

*Tính năng hệ thống ứng dụng:*

- Tính toán & Xuất dải tần điểm rủi ro: **Probability Risk Score (%)**.
- **Real-time Engine:** Kiểm tra gian lận cho mảng giao dịch cá biệt (Single Inference).
- **Batch Processing:** Nhúng file `.CSV` giao dịch để trả về dữ liệu giám định hàng loạt (Mass Scanning).

---

## 7. Hướng dẫn Cài đặt (Local Development)

**Bước 1: Clone Repository**

```bash
git clone https://github.com/kttt294/Fraud-Detection-System.git
cd Fraud-Detection-System
```

**Bước 2: Cài đặt Thư viện Phụ thuộc**

```bash
pip install -r requirements.txt
```

**Bước 3: Khởi chạy Ứng dụng Web / Web UI**

```bash
streamlit run deployment/app.py
```

**Khám phá quy trình Huấn luyện:**
Các file `.ipynb` trong thư mục chính chứa toàn bộ mã nguồn quá trình EDA, phân tách Model và AutoTunerCV.
