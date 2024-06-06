import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import train_test_split
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
    print(f"Data Loaded: {data_path}, {data['features'].shape[1]} features, {data['features'].shape[0]} observations.")
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

# Main code to replicate the Logistic Regression model
file_path = '~/GraLNA/data_FraudDetection_JAR2020.csv'
results = []


for year_test in range(2003, 2015):
    np.random.seed(0)
    print(f"==> Running Logistic Regression (training period: 1991-{year_test-2}, testing period: {year_test}, with 2-year gap)...")

    # Read training data
    data_train = data_reader(file_path, 1991, year_test - 2)
    X_train = data_train['features']
    y_train = data_train['labels']
    paaer_train = data_train['paaers']

    # Read testing data
    data_test = data_reader(file_path, year_test, year_test)
    X_test = data_test['features']
    y_test = data_test['labels']
    paaer_test = np.unique(data_test['paaers'][data_test['labels'] != 0])

    # Handle serial frauds using PAAER
    y_train[np.isin(paaer_train, paaer_test)] = 0

    # Train model
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=0)
    lr.fit(X_train, y_train)
    print("Training completed.")

    # Test model
    label_predict = lr.predict(X_test)
    dec_values = lr.predict_proba(X_test)[:, 1]
    print("Testing completed.")

    # Print performance results
    for topN in [0.01, 0.02, 0.03, 0.04, 0.05]:
        metrics = evaluate(y_test, label_predict, dec_values, topN)
        print(f"Performance (top {topN*100:.0f}% as cut-off thresh):")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"NDCG@k: {metrics['ndcg_at_k']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity_topk']*100:.2f}%")
        print(f"Precision: {metrics['precision_topk']*100:.2f}%")
        results.append((year_test, topN, metrics))

# Optionally save the results to a file
results_df = pd.DataFrame(results, columns=['year_test', 'topN', 'metrics'])
results_df.to_csv('results_logistic_regression.csv', index=False)
print("Results saved.")

year_valid = 2001
results_tuning = []

for max_iter in [100, 200, 500, 1000, 1500]:
    np.random.seed(0)
    print(f"==> Validating Logistic Regression - max_iter: {max_iter} (training period: 1991-{year_valid-2}, validating period: {year_valid}, with 2-year gap)...")

    # Read training data
    data_train = data_reader(file_path, 1991, year_valid - 2)
    X_train = data_train['features']
    y_train = data_train['labels']
    paaer_train = data_train['paaers']

    # Read validating data
    data_valid = data_reader(file_path, year_valid, year_valid)
    X_valid = data_valid['features']
    y_valid = data_valid['labels']
    paaer_valid = np.unique(data_valid['paaers'][data_valid['labels'] != 0])

    # Handle serial frauds using PAAER
    y_train[np.isin(paaer_train, paaer_valid)] = 0

    # Train model
    lr = LogisticRegression(max_iter=max_iter, class_weight='balanced', random_state=0)
    lr.fit(X_train, y_train)
    print("Training completed.")

    # Validate model
    label_predict = lr.predict(X_valid)
    dec_values = lr.predict_proba(X_valid)[:, 1]
    print("Validation completed.")

    # Print validation results
    metrics = evaluate(y_valid, label_predict, dec_values, 0.01)
    print(f"Max Iterations: {max_iter} ==> AUC: {metrics['auc']:.4f}")
    results_tuning.append((max_iter, metrics['auc']))

# Optionally save the tuning results to a file
tuning_df = pd.DataFrame(results_tuning, columns=['max_iter', 'auc'])
tuning_df.to_csv('tuning_logistic_regression.csv', index=False)
print("Tuning results saved.")
