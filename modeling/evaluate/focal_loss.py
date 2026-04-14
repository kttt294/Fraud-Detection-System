base_focal_single = FocalXGB(n_estimators=100)

param_distributions_focal_single = {
    'gamma': [0.5, 1.0, 1.5, 2.0],
    'alpha': [0.75, 0.85, 0.9, 0.95],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

focal_single_pipeline = AutoTunerCV(
    estimator=base_focal_single,
    param_distributions=param_distributions_focal_single,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=50,
    random_state=42,
    n_jobs=-1
)

focal_single_pipeline.fit(X_train, y_train)

summary_focal_single = focal_single_pipeline.get_params_summary()
print(f"Tham số Tốt nhất (Single Focal): {summary_focal_single['best_params']}")
print(f"Threshold tối ưu (Single Focal): {summary_focal_single['best_threshold']:.4f}")