from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# TUNE BASE XGBOOST
xgb_base = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, random_state=42, eval_metric='logloss', n_estimators=100)

param_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

xgb_pipeline_stacking = AutoTunerCV(
    estimator=xgb_base,
    param_distributions=param_xgb,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=5,
    random_state=42,
    n_jobs=-1
)
xgb_pipeline_stacking.fit(X_train, y_train)

best_xgb_params = xgb_pipeline_stacking.get_params_summary()['best_params']
print(f"-> Tham số Tốt nhất của XGBoost (Base): {best_xgb_params}\n")

xgb_best_model = XGBClassifier(
    scale_pos_weight=SCALE_POS_WEIGHT, random_state=42,
    eval_metric='logloss', n_estimators=100, **best_xgb_params
)

# TUNE BASE RANDOM FOREST
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)

param_rf = {
    'n_estimators': [100, 150],
    'max_depth': [10, 15, 20],
    'max_samples': [0.6, 0.8]
}

rf_pipeline_stacking = AutoTunerCV(
    estimator=rf_base,
    param_distributions=param_rf,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=5,
    random_state=42,
    n_jobs=-1
)
rf_pipeline_stacking.fit(X_train, y_train)

best_rf_params = rf_pipeline_stacking.get_params_summary()['best_params']
print(f"-> Tham số Tốt nhất của Random Forest (Base): {best_rf_params}\n")

rf_best_model = RandomForestClassifier(
    class_weight='balanced', random_state=42, n_jobs=1, **best_rf_params
)

# TUNE STACKING META-MODEL & THRESHOLD
meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')

stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_best_model), ('rf', rf_best_model)],
    final_estimator=meta_model,
    cv=2,
    n_jobs=-1
)
param_stacking = {
    'final_estimator__C': [0.01, 0.1, 1.0, 10.0],
    'final_estimator__penalty': ['l2']
}
stacking_pipeline = AutoTunerCV(
    estimator=stacking_model,
    param_distributions=param_stacking,
    n_splits=2,
    scoring='f2',
    threshold_metric='f1',
    n_iter=4,
    random_state=42,
    n_jobs=1
)
stacking_pipeline.fit(X_train, y_train)
summary_stacking = stacking_pipeline.get_params_summary()
print(f"\nTham số Hội tụ của Lõi Meta-Model (Stacking): {summary_stacking['best_params']}")
print(f"Ngưỡng chốt chặn Threshold Hội tụ Trấn phái (Stacking): {summary_stacking['best_threshold']:.4f}")