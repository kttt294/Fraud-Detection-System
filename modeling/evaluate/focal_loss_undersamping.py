from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

focal_hybrid_model = ImbPipeline([
    ('undersample', RandomUnderSampler(random_state=42)),
    ('model', FocalXGB(n_estimators=100))
])

param_distributions_focal_hybrid = {
    'undersample__sampling_strategy': [0.1, 0.3, 0.5],
    'model__gamma': [1.0, 1.5, 2.0],
    'model__alpha': [0.75, 0.9],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1]
}

focal_hybrid_pipeline = AutoTunerCV(
    estimator=focal_hybrid_model,
    param_distributions=param_distributions_focal_hybrid,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=50,
    random_state=42,
    n_jobs=-1
)

focal_hybrid_pipeline.fit(X_train, y_train)

summary_focal_hybrid = focal_hybrid_pipeline.get_params_summary()
print(f"Tham số Tốt nhất (Hybrid Focal): {summary_focal_hybrid['best_params']}")
print(f"Threshold tối ưu (Hybrid Focal): {summary_focal_hybrid['best_threshold']:.4f}")