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

# --- Section: Exploratory Data Analysis (EDA) ---
print("\n--- Generating EDA Visualizations ---")

# 1. Class Distribution (Mất cân bằng)
plt.figure(figsize=(8,6))
ax = sns.countplot(x='Class', data=df, hue='Class', palette='tab10', legend=False)
plt.title('Phân phối Class: Giao dịch Hợp lệ vs Gian lận', fontsize=14)
plt.xticks([0, 1], ['Hợp lệ (0)', 'Gian lận (1)'])
plt.ylabel('Số lượng (Count)')
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=10)
plt.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'))
plt.close()

# 2. Time Distribution (Chu kỳ 2 ngày)
plt.figure(figsize=(12,4))
sns.histplot(df['Time'], bins=48, kde=False, color='royalblue', edgecolor='none')
plt.title('Phân phối Time (Tổng cộng 48 giờ)', fontsize=14)
plt.xlabel('Thời gian - Time (Giây)')
plt.ylabel('Số lượng - Count')
plt.savefig(os.path.join(OUTPUT_DIR, '02_time_distribution.png'))
plt.close()

# 3. Amount vs Class (Scatter Plot)
plt.figure(figsize=(10,6))
plt.scatter(df['Amount'], df['Class'], alpha=0.5, c=df['Class'], cmap='coolwarm')
plt.title('Mối quan hệ giữa Amount và Class (Scatter)', fontsize=14)
plt.xlabel('Số tiền - Amount')
plt.ylabel('Phân loại - Class (0=Hợp lệ, 1=Gian lận)')
plt.savefig(os.path.join(OUTPUT_DIR, '03_amount_scatter.png'))
plt.close()

# 4. Amount Boxplot (Outliers)
plt.figure(figsize=(8,6))
sns.boxplot(x='Class', y='Amount', data=df, hue='Class', palette='muted', legend=False)
plt.title('Phân tích giá trị ngoại lệ Amount theo Class (Boxplot)', fontsize=14)
plt.xticks([0, 1], ['Hợp lệ (0)', 'Gian lận (1)'])
plt.savefig(os.path.join(OUTPUT_DIR, '04_amount_boxplot.png'))
plt.close()

# 5. Phân phối Amount (Thang đo Tuyến tính & Log)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df['Amount'], kde=True, color='steelblue', ax=ax1)
ax1.set_title('Phân phối Amount (Thang đo Tuyến tính)', fontsize=14)
ax1.set_xlabel('Số tiền - Amount')

sns.histplot(df['Amount'] + 0.001, kde=True, color='mediumseagreen', log_scale=True, ax=ax2)
ax2.set_title('Phân phối Amount (Thang đo Logarithm)', fontsize=14)
ax2.set_xlabel('Số tiền - Amount (Log scale)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_amount_distribution.png'))
plt.close()

# 6. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, 
            cmap='RdBu_r', 
            center=0, 
            annot=False, 
            linewidths=.5, 
            linecolor='black', 
            square=True,
            cbar_kws={"shrink": .8})
plt.title('Ma trận tương quan giữa các đặc trưng (Heatmap)', fontsize=16, pad=20)
plt.savefig(os.path.join(OUTPUT_DIR, '06_correlation_heatmap.png'), bbox_inches='tight')
plt.close()

# In ra các cột có tương quan cao nhất với Class để "bàn giao" cho nhóm Modeling
top_corr_features = corr['Class'].abs().sort_values(ascending=False).head(10)
print("\n--- Top 10 đặc trưng tương quan mạnh nhất với Class ---")
print(top_corr_features)

# 7. Phân phối Toàn bộ 30 Đặc trưng (Feature Distributions Overview)
print("\n--- Generating All-Feature Distribution Overview Plot ---")
features_to_plot = df.drop('Class', axis=1).columns
plt.figure(figsize=(24, 20)) 
for i, col in enumerate(features_to_plot):
    plt.subplot(6, 5, i + 1)
    sns.histplot(df[col], kde=False, color='steelblue', bins=50)
    plt.title(f'Distribution: {col}', fontsize=12)
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.suptitle('Tổng quan Phân phối của Toàn bộ 30 Đặc trưng', fontsize=20, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_all_features_distribution.png'), bbox_inches='tight')
plt.close()
print(f"[INFO] Đã lưu biểu đồ tổng quan phân phối tại '{OUTPUT_DIR}/07_all_features_distribution.png'")

# 8. Phân tích Thang đo Đặc trưng (Feature Scale Analysis)
print("\n--- Generating Feature Scale Comparison Plot ---")
stats = df.drop('Class', axis=1).agg(['mean', 'std']).T
fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(20, 8))

# --- BIỂU ĐỒ 1: THANG ĐO TUYẾN TÍNH ---
ax_lin.bar(stats.index, stats['mean'], color='steelblue', alpha=0.6, label='Mean')
ax_lin.set_ylabel('Giá trị (Thang đo thường)')
ax_lin.set_title('Thang đo Tuyến tính', fontsize=14)

ax_lin_2 = ax_lin.twinx()
ax_lin_2.plot(stats.index, stats['std'], color='red', marker='o', markersize=4, label='Std Dev')
ax_lin_2.set_ylabel('Độ lệch chuẩn (Std)')

# --- BIỂU ĐỒ 2: THANG ĐO LOGARITHM (SYMLOG) ---
ax_log.bar(stats.index, stats['mean'], color='steelblue', alpha=0.6, label='Mean')
ax_log.plot(stats.index, stats['std'], color='red', marker='o', markersize=4, label='Std Dev')
ax_log.set_yscale('symlog') 
ax_log.set_ylabel('Giá trị (Thang đo Log)')
ax_log.set_title('Thang đo Logarithm', fontsize=14)

for ax in [ax_lin, ax_log]:
    ax.set_xticks(range(len(stats.index)))
    ax.set_xticklabels(stats.index, rotation=90)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_feature_scale_analysis.png'), bbox_inches='tight')
plt.close()
print(f"[INFO] Đã lưu biểu đồ phân tích thang đo tại '{OUTPUT_DIR}/08_feature_scale_analysis.png'")

# 3 & 4. Splitting then Scaling (Best Practice to tránh Data Leakage)
print("\n--- Splitting then Scaling (Best Practice) ---")
X = df.drop('Class', axis=1)
y = df['Class']

# Tách tập Train/Test TRƯỚC khi scale để đảm bảo không rò rỉ thông tin từ tập Test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Khởi tạo Scaler
rob_scaler = RobustScaler()

# CHỈ FIT trên tập Train dữ liệu thô
train_scaled = rob_scaler.fit_transform(X_train_raw[['Amount', 'Time']])
# Chỉ transform trên tập Test dựa trên thông số Median/IQR đã học từ Train
test_scaled = rob_scaler.transform(X_test_raw[['Amount', 'Time']])

# Lưu scaler đã học để dùng lại trong deployment (Best Practice: Serving)
MODELING_DIR = os.path.join(BASE_DIR, 'modeling')
os.makedirs(MODELING_DIR, exist_ok=True)
with open(os.path.join(MODELING_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(rob_scaler, f)
print(f"[INFO] Đã lưu RobustScaler tại '{MODELING_DIR}/scaler.pkl'")

# Hàm phụ trợ để hoàn thiện DataFrame sau scale
def finalize_df(df_raw, scaled_values):
    df_new = df_raw.copy()
    df_new['scaled_amount'] = scaled_values[:, 0]
    df_new['scaled_time'] = scaled_values[:, 1]
    df_new.drop(['Amount', 'Time'], axis=1, inplace=True)
    # Đưa 2 cột đã scale lên đầu tiên cho đúng cấu trúc model yêu cầu
    cols = ['scaled_amount', 'scaled_time'] + [c for c in df_new.columns if c not in ['scaled_amount', 'scaled_time']]
    return df_new[cols]

# Tạo tập X_train và X_test chính thức
X_train = finalize_df(X_train_raw, train_scaled)
X_test = finalize_df(X_test_raw, test_scaled)

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
