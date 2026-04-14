base_focal_ensemble = FocalEnsembleXGB(n_estimators=100)

param_distributions_focal = {
    'gamma_wide': [0.5, 1.0, 1.25],
    'gamma_deep': [1.25, 1.5, 2.0],
    'ensemble_weight': [0.2, 0.4, 0.5, 0.6],
    'max_depth': [3, 6, 8, 10]
}

focal_ensemble_pipeline = AutoTunerCV(
    estimator=base_focal_ensemble,
    param_distributions=param_distributions_focal,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=50,
    random_state=42,
    n_jobs=-1
)

focal_ensemble_pipeline.fit(X_train, y_train)

summary_focal = focal_ensemble_pipeline.get_params_summary()
print(f"Tham số Tốt nhất (Self-Ensemble Focal Loss): {summary_focal['best_params']}")
print(f"Ngưỡng chốt chặn Threshold (Self-Ensemble Focal Loss): {summary_focal['best_threshold']:.4f}")