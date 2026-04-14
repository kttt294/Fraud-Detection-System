from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with open('data/processed/data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']

SCALE_POS_WEIGHT = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_focal = FocalXGB(alpha=0.9, gamma=1.25)
print(f"\nXGBoost Focal Loss (Alpha={xgb_focal.alpha}, Gamma={xgb_focal.gamma}):")
print(f"{'Fold':<7} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'F2-score':<8} | {'AUPRC':<8}")
fold_metrics = []
for i, (train_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    # Huân luyện và dự báo
    xgb_focal.fit(X_tr, y_tr)
    proba = xgb_focal.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    # Tính toán các chỉ số
    metrics = [
        precision_score(y_val, pred, zero_division=0),
        recall_score(y_val, pred),
        f1_score(y_val, pred),
        fbeta_score(y_val, pred, beta=2), # F2-Score chuẩn
        average_precision_score(y_val, proba)
    ]
    fold_metrics.append(metrics)
    print(f"Fold {i+1:<2} | {metrics[0]:<9.4f} | {metrics[1]:<8.4f} | {metrics[2]:<8.4f} | {metrics[3]:<8.4f} | {metrics[4]:<8.4f}")
ans = np.mean(fold_metrics, axis=0)
print('-'*62)
print(f"{'Average':<7} | {ans[0]:<9.4f} | {ans[1]:<8.4f} | {ans[2]:<8.4f} | {ans[3]:<8.4f} | {ans[4]:<8.4f}\n")