import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings

warnings.filterwarnings('ignore')

def data_reader(data_path, year_start, year_end):
    df = pd.read_csv(data_path)
    df = df[(df['fyear'] >= year_start) & (df['fyear'] <= year_end)]
    data = {
        'years': df['fyear'].values,
        'firms': df['gvkey'].values,
        'paaers': df['p_aaer'].values,
        'labels': df['misstate'].values,
        'features': df.drop(columns=['fyear', 'gvkey', 'p_aaer', 'misstate']).values
    }
    print(f"Data Loaded: {data_path}, {data['features'].shape[1]} features, {data['features'].shape[0]} observations for years {year_start} to {year_end}.")
    return data

def evaluate(y_true, y_pred, dec_values, topN):
    results = {}
    results['auc'] = roc_auc_score(y_true, dec_values)

    # Top N% cutoff
    k = int(len(y_true) * topN)
    top_k_idx = np.argsort(dec_values)[-k:]
    y_pred_topk = np.zeros_like(y_true)
    y_pred_topk[top_k_idx] = 1

    tp_topk = np.sum((y_true == 1) & (y_pred_topk == 1))
    fn_topk = np.sum((y_true == 1) & (y_pred_topk == 0))
    fp_topk = np.sum((y_true == 0) & (y_pred_topk == 1))
    tn_topk = np.sum((y_true == 0) & (y_pred_topk == 0))

    sensitivity_topk = tp_topk / (tp_topk + fn_topk) if (tp_topk + fn_topk) > 0 else 0
    precision_topk = tp_topk / (tp_topk + fp_topk) if (tp_topk + fp_topk) > 0 else 0

    results['sensitivity_topk'] = sensitivity_topk
    results['precision_topk'] = precision_topk

    # NDCG@k
    ndcg_at_k = ndcg_score([y_true], [dec_values], k=k)
    results['ndcg_at_k'] = ndcg_at_k

    return results

# Main code to replicate the SVM model
file_path = '~/GraLNA/data_FraudDetection_JAR2020.csv'
results = []

# Initial training and validation period
training_period_start = 1991
validation_period_end = 2001
validation_period_start = 2000

print("Reading training data...")
data_train = data_reader(file_path, training_period_start, validation_period_end)
X_train = data_train['features']
y_train = data_train['labels']
paaer_train = data_train['paaers']

# Handle missing values
print("Handling missing values in training data...")
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)

# Ensure no NaN values
if np.isnan(X_train).any():
    print("Warning: NaNs detected in training data after imputation.")
    X_train = np.nan_to_num(X_train)
    print("NaNs replaced with zeros.")

print("Reading validation data...")
data_valid = data_reader(file_path, validation_period_start, validation_period_end)
X_valid = data_valid['features']
y_valid = data_valid['labels']
paaer_valid = np.unique(data_valid['paaers'][data_valid['labels'] != 0])

# Handle missing values
print("Handling missing values in validation data...")
X_valid = imputer.transform(X_valid)

# Ensure no NaN values
if np.isnan(X_valid).any():
    print("Warning: NaNs detected in validation data after imputation.")
    X_valid = np.nan_to_num(X_valid)
    print("NaNs replaced with zeros.")

# Grid search for optimal C+ and C- parameters using validation set
print("Performing grid search for optimal C parameter...")
param_grid = {'C': [0.01, 0.1, 1, 10, 20]}
svc = SVC(probability=True, random_state=0, class_weight='balanced')
clf = GridSearchCV(svc, param_grid, scoring='roc_auc', cv=5)
clf.fit(X_train, y_train)
best_C = clf.best_params_['C']
print(f"Optimal C parameter found: {best_C}")

# Train model with best parameter on the full training set
print("Training final SVM model with optimal C parameter...")
svc = SVC(C=best_C, probability=True, random_state=0, class_weight='balanced')
svc.fit(X_train, y_train)

for year_test in range(2003, 2009):
    print(f"==> Testing SVM (training period: 1991-{year_test-2}, testing period: {year_test}, with 2-year gap)...")

    # Read testing data
    print(f"Reading testing data for year {year_test}...")
    data_test = data_reader(file_path, year_test, year_test)
    X_test = data_test['features']
    y_test = data_test['labels']

    # Handle missing values
    print("Handling missing values in testing data...")
    X_test = imputer.transform(X_test)

    # Ensure no NaN values
    if np.isnan(X_test).any():
        print("Warning: NaNs detected in testing data after imputation.")
        X_test = np.nan_to_num(X_test)
        print("NaNs replaced with zeros.")

    # Test model
    print("Making predictions with the SVM model...")
    label_predict = svc.predict(X_test)
    dec_values = svc.predict_proba(X_test)[:, 1]

    # Print performance results
    for topN in [0.01, 0.02, 0.03, 0.04, 0.05]:
        metrics = evaluate(y_test, label_predict, dec_values, topN)
        print(f"Performance (top {topN*100:.0f}% as cut-off thresh):")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"NDCG@k: {metrics['ndcg_at_k']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity_topk']*100:.2f}%")
        print(f"Precision: {metrics['precision_topk']*100:.2f}%")
        results.append((year_test, topN, metrics))

print("Saving results to a CSV file...")
results_df = pd.DataFrame(results, columns=['year_test', 'topN', 'metrics'])
results_df.to_csv('results_svm.csv', index=False)
print("Results saved.")
