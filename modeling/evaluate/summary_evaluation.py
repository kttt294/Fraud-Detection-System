import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, average_precision_score, roc_auc_score
import pickle

final_tuned_models = {
    'LightGBM CW': lgbm_pipeline,
    'XGBoost CW': xgb_pipeline_main,
    'SMOTE (0.05) + XGBoost': smote_xgb_pipeline,
    'Single Focal Loss': focal_single_pipeline,
    'Hybrid Focal Loss': focal_hybrid_pipeline,
    'Soft Voting (XGB+RF)': voting_pipeline,
    'Stacking (Meta-LR)': stacking_pipeline,
    'SMOTE (0.05) + Random Forest': rf_smote_pipeline,
    'Self-Ensemble Focal Loss': focal_ensemble_pipeline
}

test_results = []

for name, pipeline in final_tuned_models.items():
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred  = pipeline.predict(X_test)

    params_summary = pipeline.get_params_summary()
    best_thresh = params_summary['best_threshold']

    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    f2 = fbeta_score(y_test, test_pred, beta=2)
    auprc = average_precision_score(y_test, test_proba)
    roc_auc = roc_auc_score(y_test, test_proba)

    test_results.append({
        'Tên Mô Hình': name,
        'Best Threshold': round(best_thresh, 4),
        'Accuracy': round(acc, 4),
        'ROC-AUC': round(roc_auc, 4),
        'AUPRC': round(auprc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'F2-Score': round(f2, 4)
    })


if test_results:
    df_test = pd.DataFrame(test_results)
    df_test = df_test.sort_values(by='AUPRC', ascending=False).reset_index(drop=True)
    print(df_test.to_string(index=False))