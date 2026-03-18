# Implementation Plan: Credit Card Fraud Detection

Dự án này xây dựng một hệ thống phát hiện giao dịch gian lận sử dụng Machine Learning, tập trung vào việc thể hiện các kỹ thuật xử lý dữ liệu phức tạp và đánh giá mô hình chuyên sâu.

### Collect data -> EDA -> data cleaning -> feature engineering -> stratified split -> training

### 1. Khám phá & Tiền xử lý Dữ liệu (EDA & Preprocessing)

1. **EDA (Exploratory Data Analysis – phân tích khám phá dữ liệu):** là bước khám phá và hiểu dữ liệu trước khi xây dựng mô hình. Sử dụng `seaborn` và `matplotlib` để phân tích sự mất cân bằng giữa giao dịch hợp lệ và gian lận. Mục đích chính là để hiểu dữ liệu trước khi đưa vào huấn luyện:

   1. xem cấu trúc dữ liệu: số cột, số dòng, kiểu dữ liệu
   2. phát hiện missing values và outliers
   3. hiểu phân bố dữ liệu
   4. tìm mối quan hệ giữa các biến
   5. kiểm tra dữ liệu có hợp lý không
   6. VD: EDA giúp trả lời: diện tích trung bình bao nhiêu, giá nhà phân bố thế nào, diện tích và giá có tương quan không. EDA thường chiếm  **60–70% thời gian làm ML project** .
   7. Các bước EDA thường làm:

      1. xem tổng quan dữ liệu: df.head(), df.info(), df.describe()
      2. kiểm tra missing values: df.isnull().sum()
      3. xem phân bố dữ liệu: dùng biểu đồ histgram, boxplot, density plot
      4. tìm outliers: có thể dùng boxplot, Z-score, IQR
      5. phân tích tương quan: df.corr() hoặc vẽ heatmap. Mục tiêu: xem biến nào liên quan đến target và phát hiện multicollinearity
   8. EDA quan trọng vì, nếu không làm EDA thì dễ: train model với data lỗi, bị bias, có outliers phá model, feature không liên quan
   9. EDA phải kiểm tra xem features nào trực tiếp tiết lộ target, ví dụ predict customer churn nhưng trong dataset có cột account_closed_date thì ko được.
   10. Khi làm EDA cho dataset lớn (hơn 1 triệu dòng), mục tiêu không phải là vẽ thật nhiều biểu đồ mà là cấu trúc dữ liệu NHANH và phát hiện vấn đề quan trọng. Checklist chuẩn:
   11. + kiểm tra cấu trúc dữ liệu trước: hiểu dataset có gì: bao nhiêu rows và columns, kiểu dữ liệu của mỗi cột, cột nào là categorical / numerical / datetime
       + sampling dữ liệu: dataset lớn thì không nên vẽ biểu đồ trực tiếp, thay vào đó sample_df = df.sample(100_000) -> lấy 1% hoặc 100_000 dòng sau đó làm EDA trên sample -> nhanh, tiết kiệm RAM, pattern thường vẫn giống
       + sau đó tiến hành như bình thường
2. **Handling Missing Values:** Sử dụng `KNNImputer` hoặc `IterativeImputer` thay vì Mean/Median đơn giản. *Qua thăm dò dữ liệu (EDA), tập dữ liệu Credit Card Fraud không ghi nhận giá trị thiếu (Missing values). Tuy nhiên, trong quy trình xây dựng Machine Learning Pipeline vẫn thiết lập sẵn các bước kiểm tra. Nếu trong tương lai dữ liệu đầu vào có biến động, các kỹ thuật như **KNN Imputer** hoặc **Iterative Imputer** sẽ được cân nhắc sử dụng thay vì Mean/Median để đảm bảo tính tương quan giữa các đặc trưng ẩn danh (V1-V28) không bị phá vỡ. **Tại sao dùng Imputer cao cấp cho dữ liệu PCA (V1-V28)?** Các đặc trưng V1-V28 là kết quả của phép chiếu toán học (PCA) nhằm giữ lại các mối quan hệ tuyến tính giữa các biến gốc. Việc sử dụng **KNN Imputer** hoặc **Iterative Imputer** giúp giá trị được điền vào tuân thủ **Phân phối đa biến (Multivariate Distribution)** của tập dữ liệu. Điều này đảm bảo rằng các điểm dữ liệu được điền mới không trở thành "Outliers toán học", giúp các mô hình nhạy cảm như SVM hay XGBoost đạt hiệu năng ổn định hơn. Hãy tưởng tượng bạn có 2 cột: **Chiều cao** và  **Cân nặng** . Hai cột này có tương quan thuận rất mạnh (người cao thường nặng hơn). Nếu một người bị thiếu Cân nặng, nhưng bạn lại điền bằng  **Cân nặng trung bình của cả thế giới** , con số đó sẽ hoàn toàn không khớp với Chiều cao thực tế của người đó. H**ệ quả:** Bạn đã tạo ra một "điểm dữ liệu giả" vô lý (ví dụ: cao 2m nhưng nặng 50kg). Mối quan hệ toán học giữa các cột bị phá vỡ. Máy sẽ bị rối khi học các điểm dữ liệu mâu thuẫn như vậy.
3. **Scaling:** Áp dụng `RobustScaler` vì dữ liệu tài chính thường chứa nhiều Outliers. Nếu dùng `StandardScaler`, kết quả sẽ rất tệ. Ta cũng **không nên xóa** những Outliers này, vì trong ngành ngân hàng, giao dịch bất thường (Outlier) chính là dấu hiệu tiềm năng của gian lận. RonustScaler dùng để scale các feature nhưng ít bị ảnh hưởng bởi outliers, thường dựa trên median và IQR = Q3 - Q1 thay vì mean và std như standardScaler

   - x(scaled) = (**x**−median) / **I**QR
   - Tại sao chúng ta NÊN scale cột `Time`?
     - **Sự áp đảo về con số** : Cột `Time` lên tới hơn  **170.000** , trong khi các cột `V1-V28` chỉ loanh quanh số lẻ (ví dụ 0.5, -1.2). Nếu bạn dùng các model nhạy cảm với thang đo như  **Logistic Regression, SVM, hay Neural Networks** , model sẽ lầm tưởng rằng `Time` quan trọng gấp... 100.000 lần các biến khác chỉ vì con số của nó lớn hơn. Điều này làm model bị "mù tạng" (biased).
     - **Đồng bộ hóa (Feature Engineering)** : Bằng cách dùng `RobustScaler`, chúng ta đưa `Time` về cùng một "mặt bằng" với các biến PCA (V1-V28). Model sẽ học được là: *"À, lúc này là sớm hay muộn trong ngày"* mà không bị choáng ngợp bởi con số hàng chục ngàn.
     - **Giữ nguyên thông tin** : Scale không làm mất đi thứ tự thời gian, nó chỉ "co" lại cho vừa vặn.
4. **Imbalanced Data:** Sử dụng `SMOTE` (Synthetic Minority Over-sampling Technique) để cân bằng dữ liệu. Không *SMOTE trong tiền xử lý mà đưa vào Pipeline để tránh Data Leakage trong quá trình Cross-Validation*
5. *Ma trận tương quan giúp xác định được 'mối liên hệ nhân quả' tiềm ẩn giữa các đặc trưng PCA bí ẩn và hành vi gian lận. Nhờ nó, biết được những biến nào là 'Input chất lượng' nhất cho mô hình, đồng thời kiểm chứng lại tính hiệu quả của bước PCA khi các biến đầu vào hoàn toàn độc lập với nhau, giúp mô hình huấn luyện nhanh và chính xác hơn.*
6. **Lợi thế của Tập dữ liệu đã qua xử lý PCA và Tính Thực tế trong Doanh nghiệp**

   Việc lựa chọn tập dữ liệu có các cột từ **V1 đến V28** (đã qua biến đổi PCA) thay vì tìm kiếm các tập dữ liệu thô (raw data) mang lại ba ưu thế chiến lược, đồng thời mô phỏng chính xác môi trường làm việc tại các doanh nghiệp tài chính hiện nay:

   1. **Tính bảo mật và Tuân thủ (Data Privacy & Compliance):** Trong thực tế, các ngân hàng và tổ chức tài chính không bao giờ cung cấp dữ liệu thô chứa thông tin nhạy cảm như *Tên khách hàng, Số tài khoản, hay Địa chỉ giao dịch* cho đội ngũ phân tích dữ liệu bên thứ ba hoặc thậm chí là nhân viên nội bộ nếu không cần thiết. Việc sử dụng các biến số đã được PCA hóa chính là cách các doanh nghiệp bảo vệ quyền riêng tư của khách hàng mà vẫn cho phép khai thác giá trị từ dữ liệu. Lựa chọn tập dữ liệu này giúp chúng ta làm quen với quy trình xử lý dữ liệu thực tế: **Làm việc với các đặc trưng cốt lõi thay vì các nhãn tên trực quan.**
   2. **Tối ưu hóa Hiệu năng và Khử nhiễu (Efficiency & De-noising):** Dữ liệu thô thường chứa rất nhiều biến số có tính tương quan chéo cao hoặc mang thông tin thừa (nhiễu). PCA đã thực hiện thay chúng ta bước "tinh lọc" ban đầu: nén các thông tin quan trọng nhất vào 28 thành phần chính, giúp loại bỏ các biến lặp lại và giảm đáng kể chi phí tính toán. Điều này giúp mô hình tập trung vào việc học các **mẫu hình hành vi gian lận (Fraud Patterns)** thực sự thay vì bị xao nhãng bởi các thông tin định danh không liên quan.
   3. **Mô phỏng Bài toán Doanh nghiệp Thực tế:** Tại một trung tâm quản lý rủi ro thẻ tín dụng, thách thức thực sự không nằm ở việc đọc tên cột dữ liệu là gì, mà là **khả năng phát hiện các "dấu vân tay" bất thường** ẩn sâu trong các chỉ số kỹ thuật. Việc sử dụng tập dữ liệu này buộc chúng ta phải dựa hoàn toàn vào toán học, thống kê và cấu trúc phân phối của dữ liệu để đưa ra dự đoán – đây chính là kỹ năng quan trọng nhất của một chuyên gia Data Science khi đối mặt với các hệ thống dữ liệu khổng lồ và phức tạp trong doanh nghiệp.

   **Kết luận:** Tập dữ liệu này không chỉ là một bài tập học thuật, mà là một ví dụ điển hình về cách thức **Quản trị rủi ro hiện đại** được vận hành: Bảo mật tuyệt đối, nhưng vẫn đảm bảo độ chính xác tối đa trong nhận diện gian lận.

### 2. Xây dựng & Sàng lọc Mô hình (Modeling & Spot-checking)

- có thể load data đã được chia với lệnh:
- import pickle
  with open('../data/processed/data_splits.pkl', 'rb') as f:
  data = pickle.load(f)
  X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
- **Baseline:** Logistic Regression.
- **Advanced Models:** Random Forest, XGBoost, CatBoost.
- **Quy trình Huấn luyện:** Huấn luyện nhanh các mô hình với tham số mặc định (Default) bằng **Cross-Validation 5-fold**.

  - **Chi tiết kỹ thuật:** Chia tập Train thành 5 phần (folds). Thực hiện lặp 5 lần; mỗi lần huấn luyện mô hình **hoàn toàn mới từ đầu** trên 4 phần và kiểm tra trên 1 phần còn lại.
  - **Kết quả:** Lấy trung bình cộng điểm số của 5 lần chạy để làm "chứng chỉ năng lực" cho thuật toán.
  - **Mục tiêu:** Đánh giá độ ổn định và khả năng tổng quát hóa của mô hình, từ đó tìm ra "Shortlist" (2-3 mô hình tiềm năng nhất).
- **Pipeline:** Sử dụng `sklearn.pipeline.Pipeline` kết hợp Preprocessing (Scaling + SMOTE) và Modeling để tránh Data Leakage.

### 3. Tối ưu & Đánh giá Chuyên sâu (Optimization & Evaluation)

> **Quy tắc quan trọng:** Tập **Test** chỉ được sử dụng DUY NHẤT một lần ở cuối bước 3 để báo cáo kết quả cuối cùng. Tuyệt đối không dùng tập Test để chọn mô hình hoặc chỉnh tham số.

- **Hyperparameter Tuning:** Thực hiện `RandomizedSearchCV` cho các mô hình trong Shortlist dựa trên tập **Train**.
- **Final Selection:** So sánh các mô hình ĐÃ tối ưu (dựa trên kết quả Cross-validation trên tập Train) để chọn ra mô hình cuối cùng.
- **Metrics:** Tập trung vào **Precision-Recall Curve (AUPRC)**.
- **Final Evaluation:** Chạy mô hình tốt nhất lên tập **Test** để xem hiệu suất thực tế.

### 4. Giải thích & Lưu trữ (Interpretation & Export)

* **Interpretability:** Sử dụng `SHAP` values để giải thích lý do mô hình dự đoán gian lận cho từng giao dịch cụ thể.
* **Model Export:** Lưu mô hình đã huấn luyện (Pipeline hoàn chỉnh) thành file `.pkl` để sử dụng cho Web App.

### Ứng dụng Web (Web Application - Streamlit)

* **SafeGuard Banking Dashboard:** Xây dựng giao diện Web cho phép:
  * Nhập thông số giao dịch hoặc upload CSV để kiểm tra gian lận.
  * Hiển thị cảnh báo đỏ nếu phát hiện gian lận.
  * Trực quan hóa lý do mô hình đưa ra quyết định (Explainable AI).

Nhóm xây dựng một **Hệ thống AI tích hợp (Embedded AI System)** chuyên biệt cho hạ tầng tài chính. Hệ thống được thiết kế theo kiến trúc  **Privacy-Preserving Machine Learning** .

* **Phía Ngân hàng:** Giữ bí mật về ý nghĩa các đặc trưng gốc.
* **Phía Kỹ sư (Chúng em):** Xây dựng mô hình dựa trên các đặc trưng nén (PCA) để bảo mật.
* **Phía Vận hành:** Dashboard này cung cấp giao diện trực quan để lọc ra 0.17% giao dịch gian lận từ hàng triệu giao dịch mỗi ngày giúp tối ưu hóa nguồn lực nhân sự.

*Để đảm bảo luồng dữ liệu giữa các phòng ban ngân hàng diễn ra thông suốt, hệ thống cung cấp file Template chuẩn để bộ phận IT có thể mapping dữ liệu vào đúng form.*

=> **Giải pháp Công nghệ Tài chính (FinTech Solution) -> Hệ thống hỗ trợ chuyển viên phòng rủi ro**

 *"V1 đến V28 là gì?"*

> *"Trong thực tiễn ngân hàng, đây là các **đặc trưng hành vi và thiết bị** đã được mã hóa. Ví dụ V1 có thể là chỉ số về vị trí địa lý, V2 là thông tin về loại thiết bị. Ngân hàng không cung cấp tên gốc của các cột này cho nhóm phát triển để bảo vệ quyền riêng tư của khách hàng, nhưng mô hình vẫn có thể tìm ra các mẫu hình gian lận từ những con số đã mã hóa này*



VỀ NHƯỢC ĐIỂM CỦA CÁC CỘT DỮ LIỆU Vi ĐÃ QUA PCA:

1. **Chấp nhận** : Đây là đặc thù của dữ liệu tài chính (đã được mã hóa qua PCA) nên không thể áp dụng chéo trực tiếp.
2. **Lập luận** : "Train lại" thực chất là một lợi thế vì nó tạo ra **Customized Model** sát với hành vi khách hàng của từng doanh nghiệp nhất, giúp giảm tối đa thiệt hại tài chính do báo động giả.
3. **Tầm nhìn** : Sản phẩm SafeGuard được thiết kế dưới dạng một **Pipeline tự động** (như Notebook em đã xây dựng), giúp việc "tái huấn luyện" cho khách hàng mới diễn ra cực kỳ nhanh chóng và chuẩn xác.

### Bối cảnh Dự án: Bài toán từ Doanh nghiệp

Dự án này được xây dựng dựa trên một yêu cầu thực tế từ một  **Tổ chức Tài chính quốc tế** . Bài toán đặt ra là: Doanh nghiệp đang đối mặt với nguy cơ gian lận giao dịch ngày càng tinh vi, gây thất thoát hàng triệu USD mỗi năm. Họ cần một giải pháp AI tự động hóa việc sàng lọc giao dịch nhưng phải đảm bảo các điều kiện sau:

#### - Thách thức về Bảo mật Dữ liệu (Dữ liệu PCA)

Do tính chất bảo mật nghiêm ngặt của ngành ngân hàng, doanh nghiệp không cung cấp thông tin thô của khách hàng (như tên, địa chỉ, số thẻ). Thay vào đó, dữ liệu được cung cấp dưới dạng các đặc trưng đã được **PCA (Principal Component Analysis)** — biến đổi thành 28 cột ẩn danh từ  **V1 đến V28** .

* **Nhiệm vụ của chúng ta:** Xây dựng một mô hình học máy đủ mạnh để nhận diện quy luật gian lận ngay cả trên những không gian vector ẩn danh này.

#### - Thách thức về Dữ liệu mất cân bằng (Imbalance Data)

Trong dữ liệu thực tế mà khách hàng cung cấp, các giao dịch gian lận chỉ chiếm tỉ lệ cực nhỏ ( **~0.17%** ). Nếu xử lý không khéo, mô hình sẽ mặc định coi "tất cả là hợp lệ" và bỏ lọt những kẻ tấn công thực sự.

* **Giải pháp của chúng ta:** Áp dụng kỹ thuật tiền xử lý chuyên sâu, bao gồm **Robust Scaling** (chuẩn hóa chống outliers) và chiến lược cân bằng dữ liệu **SMOTE** để huấn luyện AI nhạy bén hơn với các dấu hiệu nhỏ nhất.

#### - Yêu cầu về Triển khai Thực tế (Deployment)

Doanh nghiệp không chỉ cần một báo cáo hay một đoạn code chạy trên máy tính. Họ cần một  **Hệ thống có khả năng tương tác** :

* Một **API chuyên dụng** để các ứng dụng Mobile/Web của ngân hàng có thể gọi xác thực ngay lập tức.
* Một **Dashboard Dashboard trực quan** để đội ngũ kỹ thuật có thể theo dõi "sức khỏe" của dòng chảy giao dịch trong thời gian thực.
* Một **Hệ thống hỗ trợ chuyên viên (Specialist Assistant)** giúp xử lý các tệp dữ liệu lớn hàng chục nghìn dòng một cách tập trung.

Mục tiêu: Hệ thống **SafeGuard** được thiết kế và phát triển chính là câu trả lời cho bài toán này. Nó chuyển đổi những bộ dữ liệu số khô khan thành hành động bảo mật cụ thể, giúp doanh nghiệp chủ động ngăn chặn gian lận thay vì chỉ thụ động khắc phục hậu quả.


## Verification Plan

### Automated Tests

- Kiểm tra tính đúng đắn của Pipeline (không bị lỗi khi `fit` và `predict`).
- Kiểm tra tỷ lệ gian lận sau khi áp dụng SMOTE.
- So sánh hiệu suất giữa Baseline và các mô hình nâng cao qua các chỉ số F1, AUC-ROC.

### Manual Verification

- Kiểm tra các biểu đồ EDA để đảm bảo tính logic của dữ liệu.
- Phân tích danh sách các đặc trưng quan trọng nhất (Feature Importance) xem có phù hợp với thực tế tài chính không.
