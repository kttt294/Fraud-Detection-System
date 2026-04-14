from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

rf_smote_model = ImbPipeline([
    ('smote', SMOTE(sampling_strategy=0.05, random_state=42)),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1))
])

param_dist_rf_pipeline = {
    'model__n_estimators': [100, 150],
    'model__max_depth': [10, 15, 20],
    'model__min_samples_split': [5, 10],
    'model__min_samples_leaf': [2, 5],
    'model__max_samples': [0.3, 0.5, 0.7]
}

rf_smote_pipeline = AutoTunerCV(
    estimator=rf_smote_model,
    param_distributions=param_dist_rf_pipeline,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=10,
    random_state=42,
    n_jobs=-1
)
rf_smote_pipeline.fit(X_train, y_train)

summary_rfsmote = rf_smote_pipeline.get_params_summary()
print(f"Best Params (SMOTE + RF): {summary_rfsmote['best_params']}")
print(f"Best Threshold (SMOTE + RF): {summary_rfsmote['best_threshold']:.4f}")