from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    scale_pos_weight=SCALE_POS_WEIGHT,
    random_state=42,
    verbose=-1,
    n_estimators=100
)

param_distributions_lgbm = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [31, 50, 100],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

lgbm_pipeline = AutoTunerCV(
    estimator=lgbm_model,
    param_distributions=param_distributions_lgbm,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=50,
    random_state=42,
    n_jobs=-1
)

lgbm_pipeline.fit(X_train, y_train)

summary_lgbm = lgbm_pipeline.get_params_summary()
print(f"Tham số Tốt nhất (LightGBM CW): {summary_lgbm['best_params']}")
print(f"Ngưỡng chốt chặn Threshold (LightGBM CW): {summary_lgbm['best_threshold']:.4f}")