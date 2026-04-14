from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

smote_xgb_model = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.05, random_state=42)),
    ('model', XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, random_state=42, eval_metric='logloss', n_estimators=100))
])

param_distributions_smote_xgb = {
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 10],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

smote_xgb_pipeline = AutoTunerCV(
    estimator=smote_xgb_model,
    param_distributions=param_distributions_smote_xgb,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=50,
    random_state=42,
    n_jobs=-1
)

smote_xgb_pipeline.fit(X_train, y_train)

summary_smote_xgb = smote_xgb_pipeline.get_params_summary()
print(f"Tham số Tốt nhất (SMOTE + XGBoost): {summary_smote_xgb['best_params']}")
print(f"Ngưỡng chốt chặn Threshold (SMOTE + XGBoost): {summary_smote_xgb['best_threshold']:.4f}")