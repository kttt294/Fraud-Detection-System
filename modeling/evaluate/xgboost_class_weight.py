from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    scale_pos_weight=SCALE_POS_WEIGHT,
    random_state=42,
    eval_metric='logloss',
    n_estimators=100
)

param_distributions_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_pipeline_main = AutoTunerCV(
    estimator=xgb_model,
    param_distributions=param_distributions_xgb,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=50,
    random_state=42,
    n_jobs=-1
)

xgb_pipeline_main.fit(X_train, y_train)

summary_xgb = xgb_pipeline_main.get_params_summary()
print(f"Tham số Tốt nhất (XGBoost CW): {summary_xgb['best_params']}")
print(f"Ngưỡng chốt chặn Threshold (XGBoost CW): {summary_xgb['best_threshold']:.4f}")