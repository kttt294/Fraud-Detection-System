from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with open('data/processed/data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']

SCALE_POS_WEIGHT = y_train.value_counts()[0] / y_train.value_counts()[1]



xgb_focal_hybrid = FocalXGB(alpha=0.9, gamma=2.0)
print(f"\nXGBoost Hybrid (Undersample 50% Class 0 + Focal Loss):")
print(f"{'Fold':<7} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'F2-score':<8} | {'AUPRC':<8}")
fold_metrics = []

for i, (train_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    num_neg = np.sum(y_tr == 0)
    num_pos = np.sum(y_tr == 1)

    # Giảm lớp 0 xuống còn 50% số lượng mẫu ban đầu, giữ nguyên lớp 1
    target_strategies = {0: int(0.5 * num_neg), 1: int(num_pos)}
    rus = RandomUnderSampler(sampling_strategy=target_strategies, random_state=42)
    X_res, y_res = rus.fit_resample(X_tr, y_tr)

    xgb_focal_hybrid.fit(X_res, y_res)
    proba = xgb_focal_hybrid.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = [
        precision_score(y_val, pred, zero_division=0),
        recall_score(y_val, pred),
        f1_score(y_val, pred),
        f2_score(y_val, pred),
        average_precision_score(y_val, proba)
    ]
    fold_metrics.append(metrics)
    print(f"Fold {i+1:<2} | {metrics[0]:<9.4f} | {metrics[1]:<8.4f} | {metrics[2]:<8.4f} | {metrics[3]:<8.4f} | {metrics[4]:<8.4f}")

ans = np.mean(fold_metrics, axis=0)
print('-'*62)
print(f"{'Average':<7} | {ans[0]:<9.4f} | {ans[1]:<8.4f} | {ans[2]:<8.4f} | {ans[3]:<8.4f} | {ans[4]:<8.4f}\n")