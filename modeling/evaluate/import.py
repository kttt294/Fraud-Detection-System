from sklearn.model_selection import StratifiedKFold
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with open('data/processed/data_splits.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']

SCALE_POS_WEIGHT = y_train.value_counts()[0] / y_train.value_counts()[1]