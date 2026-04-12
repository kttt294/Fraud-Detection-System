from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import xgboost as xgb
import numpy as np
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with open('data/processed/data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']

SCALE_POS_WEIGHT = y_train.value_counts()[0] / y_train.value_counts()[1]



base_estimators = [
    ('xgb_cw', XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, random_state=42, n_jobs=-1, tree_method='hist')),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, max_samples=0.8, min_samples_leaf=5, random_state=42, n_jobs=-1))
]

ensembles = {
    'Soft Voting (XGB+RF)': VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1),
    'Stacking (Meta: Logistic)': StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1, passthrough=False)
}

for name, model in ensembles.items():
    print(f"\n{name}:")
    print(f"{'Fold':<7} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'F2-score':<8} | {'AUPRC':<8}")
    fold_metrics = []

    for i, (train_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)

        proba = m.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)

        metrics = [
            precision_score(y_val, pred, zero_division=0),
            recall_score(y_val, pred),
            f1_score(y_val, pred),
            f2_score(y_val, pred),
            average_precision_score(y_val, proba),
        ]
        fold_metrics.append(metrics)
        print(f"Fold {i+1:<2} | {metrics[0]:<9.4f} | {metrics[1]:<8.4f} | {metrics[2]:<8.4f} | {metrics[3]:<8.4f} | {metrics[4]:<8.4f}")

    ans = np.mean(fold_metrics, axis=0)
    print('-' * 62)
    print(f"{'Average':<7} | {ans[0]:<9.4f} | {ans[1]:<8.4f} | {ans[2]:<8.4f} | {ans[3]:<8.4f} | {ans[4]:<8.4f}\n")