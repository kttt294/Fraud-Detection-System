import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư mục lưu kết quả
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'outputs')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. Load Dataset
print("--- Loading Dataset ---")

file_path = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')

# Kiểm tra file sự tồn tại của file CSV
if not os.path.exists(file_path):
    print(f"[ERROR] Không tìm thấy file tại '{file_path}'.")
    print("Vui lòng giải nén file creditcard.zip vào thư mục data/raw/")
    exit()

print(f"[INFO] Bắt đầu đọc dữ liệu từ: {file_path}")
df = pd.read_csv(file_path)

# 2. EDA Cơ bản
print("\n--- Basic Information ---")
print(df.info())

missing = df.isnull().sum().sum()
print(f"\n--- Missing Values ---\nTổng số giá trị thiếu: {missing}")

class_counts = df['Class'].value_counts()
class_pct = df['Class'].value_counts(normalize=True)
print("\n--- Target Distribution (Class) ---")
print(class_counts)
print(class_pct)

# Phân tích Outliers bằng IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outlier_counts = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("\n--- Top 10 Columns With Most Outliers (IQR method) ---")
print(outlier_counts.sort_values(ascending=False).head(10))

# Lưu báo cáo EDA ra file text
with open(os.path.join(OUTPUT_DIR, 'eda_report.txt'), 'w', encoding='utf-8') as f:
    f.write(f"=== EDA REPORT ===\n")
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Missing values: {missing}\n\n")
    f.write(f"Class Distribution (count):\n{class_counts.to_string()}\n\n")
    f.write(f"Class Distribution (percent):\n{class_pct.to_string()}\n\n")
    f.write(f"Amount Statistics:\n{df['Amount'].describe().to_string()}\n\n")
    f.write(f"Top Outlier Columns (IQR method):\n{outlier_counts.sort_values(ascending=False).head(10).to_string()}\n")
print(f"\n[INFO] Đã lưu báo cáo EDA tại '{OUTPUT_DIR}/eda_report.txt'")

# 3. Visualization nâng cao: Ma trận Tương quan (Correlation Heatmap)
print("\n--- Generating Correlation Heatmap ---")
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
plt.title('Ma trận Tương quan (Gợi ý đặc trưng quan trọng)', fontsize=14)
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
plt.close()

# In ra các cột có tương quan cao nhất với Class để "bàn giao" cho nhóm Modeling
top_corr_features = corr['Class'].abs().sort_values(ascending=False).head(10)
print("\n--- Top 10 đặc trưng tương quan mạnh nhất với Class ---")
print(top_corr_features)

# Visualization 1: Phân phối của Class (Mất cân bằng)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Class', data=df, hue='Class', palette='viridis', legend=False)
plt.title('Phân phối Giao dịch: Hợp lệ (0) vs Gian lận (1) (Log Scale)')

# Thêm số lượng chính xác trên đầu mỗi cột
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=11, fontweight='bold')

# Sử dụng thang đo Log để có thể nhìn thấy cột của lớp 1
plt.yscale('log') 
plt.ylabel('Số lượng (Log Scale)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_class.png'))
plt.close()
print(f"\n[INFO] Đã lưu biểu đồ phân phối Class tại '{OUTPUT_DIR}/distribution_class.png'")

# Visualization 2a: Phân phối của Amount (Dòng tiền - Thang đo Tuyến tính)
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], kde=True, color='steelblue')
plt.title('Phân phối Số tiền Giao dịch (Linear Scale)')
plt.xlabel('Số tiền (Euro)', fontsize=12)
plt.ylabel('Số lượng giao dịch', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_amount.png'))
plt.close()
print(f"[INFO] Đã lưu biểu đồ phân phối Amount (Linear) tại '{OUTPUT_DIR}/distribution_amount.png'")

# Visualization 2b: Phân phối của Amount (Dòng tiền - Thang đo Logarithm)
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'] + 0.001, kde=True, color='mediumseagreen', log_scale=True)
plt.title('Phân phối của Số tiền Giao dịch (Log Scale X)')
plt.xlabel('Số tiền (Log Scale)')
plt.ylabel('Số lượng')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_amount_log.png'))
plt.close()
print(f"[INFO] Đã lưu biểu đồ phân phối Amount (Log) tại '{OUTPUT_DIR}/distribution_amount_log.png'")

# Visualization 3: Boxplot Amount theo Class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette='Set2', legend=False)
plt.title('So sánh Số tiền Giao dịch theo Loại (0 vs 1)', fontsize=14)
plt.xlabel('Loại giao dịch (0: Hợp lệ, 1: Gian lận)', fontsize=12)
plt.ylabel('Số tiền (Euro)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_amount_by_class.png'))
plt.close()
print(f"[INFO] Đã lưu boxplot Amount theo Class tại '{OUTPUT_DIR}/boxplot_amount_by_class.png'")

# 3. Preprocessing: Scaling (Time & Amount)
# Sử dụng RobustScaler vì dữ liệu tài chính thường có Outliers lớn
print("\n--- Scaling Time and Amount ---")
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Loại bỏ cột gốc sau khi đã scale
df.drop(['Time','Amount'], axis=1, inplace=True)

# Đưa các cột đã scale lên đầu để dễ quan sát
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']
df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# 4. Tách tập Train và Test (Lưu ý: Không chạm vào Test cho đến bước cuối)
print("\n--- Splitting Train and Test sets ---")
X = df.drop('Class', axis=1)
y = df['Class']

# Stratify=y để đảm bảo tỷ lệ Class 0/1 đồng đều ở cả 2 tập
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Tỷ lệ gian lận trong tập Train: {y_train.mean():.4%}")
print(f"Tỷ lệ gian lận trong tập Test: {y_test.mean():.4%}")

# 5. Lưu lại dữ liệu đã tiền xử lý
print("\n--- Saving Preprocessed Data ---")

# Lưu dạng CSV
X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False, header=True)
y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False, header=True)
print(f"[INFO] Đã lưu X_train, X_test, y_train, y_test dạng CSV vào thư mục '{PROCESSED_DIR}/'")

# Lưu dạng pickle (load nhanh hơn nhiều trong các bước tiếp theo)
with open(os.path.join(PROCESSED_DIR, 'data_splits.pkl'), 'wb') as f:
    pickle.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, f)
print(f"[INFO] Đã lưu toàn bộ data splits dạng pickle tại '{PROCESSED_DIR}/data_splits.pkl'")

print("\n--- Bước 1 hoàn tất! Sẵn sàng cho Bước 2: Modeling & Spot-checking ---")
print(f"Tất cả kết quả được lưu trong thư mục: '{OUTPUT_DIR}/'")
