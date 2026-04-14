from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

xgb_base = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, random_state=42, eval_metric='logloss')
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)

voting_model = VotingClassifier(
    estimators=[('xgb', xgb_base), ('rf', rf_base)],
    voting='soft',
    n_jobs=1
)

param_distributions_voting = {
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__max_depth': [3, 5, 7],
    'xgb__subsample': [0.8, 1.0],
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 15],
    'rf__max_samples': [0.6, 0.8, 1.0],
    'weights': [[1, 1], [2, 1], [1, 2]]
}

voting_pipeline = AutoTunerCV(
    estimator=voting_model,
    param_distributions=param_distributions_voting,
    n_splits=3,
    scoring='f2',
    threshold_metric='f1',
    n_iter=10,
    random_state=42,
    n_jobs=-1
)

voting_pipeline.fit(X_train, y_train)

summary_voting = voting_pipeline.get_params_summary()
print(f"Tham số Tốt nhất (Voting XGB+RF): {summary_voting['best_params']}")
print(f"Ngưỡng chốt chặn Threshold (Voting XGB+RF): {summary_voting['best_threshold']:.4f}")