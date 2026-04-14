import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
with open('data/processed/data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X_train']
y = data['y_train']

results_kb3 = {}

models_kb3 = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

for name, model in models_kb3.items():
    print(f"\n{name}:")
    print(f"{'Fold':<7} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'F2-score':<8} | {'AUPRC':<8}")

    fold_metrics = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = deepcopy(model)
        if name in ['XGBoost', 'LightGBM']:
            spw = y_tr.value_counts()[0] / y_tr.value_counts()[1]
            m.set_params(scale_pos_weight=spw)

        m.fit(X_tr, y_tr)

        pred = m.predict(X_val)
        proba = m.predict_proba(X_val)[:, 1]

        metrics = [
            precision_score(y_val, pred, zero_division=0),
            recall_score(y_val, pred, zero_division=0),
            f1_score(y_val, pred, zero_division=0),
            f2_score(y_val, pred),
            average_precision_score(y_val, proba)
        ]
        fold_metrics.append(metrics)
        print(f"Fold {i+1:<2} | {metrics[0]:<9.4f} | {metrics[1]:<8.4f} | {metrics[2]:<8.4f} | {metrics[3]:<8.4f} | {metrics[4]:<8.4f}")

    ans = np.mean(fold_metrics, axis=0)
    print('-' * 62)
    print(f"{'Average':<7} | {ans[0]:<9.4f} | {ans[1]:<8.4f} | {ans[2]:<8.4f} | {ans[3]:<8.4f} | {ans[4]:<8.4f}\n")

    results_kb3[name] = {
        'Recall': ans[1],
        'AUPRC': ans[4],
        'F1': ans[2],
        'F2': ans[3]
    }

models = list(results_kb3.keys())
recall_scores = [results_kb3[m]['Recall'] for m in models]
auprc_scores = [results_kb3[m]['AUPRC'] for m in models]
f1_scores = [results_kb3[m]['F1'] for m in models]
f2_scores = [results_kb3[m]['F2'] for m in models]

# 4 màu tương ứng đúng với 4 model
model_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231']  # đỏ, lục, xanh, cam
palette = {m: model_colors[i] for i, m in enumerate(models)}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('So sánh các mô hình - Phương án 3 (Trọng số mất cân bằng)', fontsize=16, fontweight='bold')

# Vẽ Recall
sns.barplot(x=models, y=recall_scores, ax=axes[0, 0], palette=palette, hue=models, legend=False)
axes[0, 0].set_title('Recall', fontsize=14)
axes[0, 0].tick_params(axis='x', rotation=45)

# Vẽ AUPRC
sns.barplot(x=models, y=auprc_scores, ax=axes[0, 1], palette=palette, hue=models, legend=False)
axes[0, 1].set_title('AUPRC', fontsize=14)
axes[0, 1].tick_params(axis='x', rotation=45)

# Vẽ F1-score
sns.barplot(x=models, y=f1_scores, ax=axes[1, 0], palette=palette, hue=models, legend=False)
axes[1, 0].set_title('F1-Score', fontsize=14)
axes[1, 0].tick_params(axis='x', rotation=45)

# Vẽ F2-score
sns.barplot(x=models, y=f2_scores, ax=axes[1, 1], palette=palette, hue=models, legend=False)
axes[1, 1].set_title('F2-Score', fontsize=14)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()