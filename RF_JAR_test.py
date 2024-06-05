import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import time

def data_reader(data_path, data_type, year_start, year_end):
    temp = pd.read_csv(data_path).values
    data = {}
    if data_type == 'default':
        data['years'] = temp[:, 0]
        idx = (data['years'] >= year_start) & (data['years'] <= year_end)
        data['years'] = temp[idx, 0]
        data['firms'] = temp[idx, 1]
        data['labels'] = temp[idx, 2].astype(int)
        data['features'] = temp[idx, 3:]
        data['num_obervations'] = data['features'].shape[0]
        data['num_features'] = data['features'].shape[1]
    elif data_type == 'uscecchini28':
        data['years'] = temp[:, 0]
        idx = (data['years'] >= year_start) & (data['years'] <= year_end)
        data['years'] = temp[idx, 0]
        data['firms'] = temp[idx, 1]
        data['sics'] = temp[idx, 2]
        data['insbnks'] = temp[idx, 3]
        data['understatements'] = temp[idx, 4]
        data['options'] = temp[idx, 5]
        data['paaers'] = temp[idx, 6]
        data['newpaaers'] = temp[idx, 7]
        data['labels'] = temp[idx, 8].astype(int)
        data['features'] = temp[idx, 9:37]
        data['num_obervations'] = data['features'].shape[0]
        data['num_features'] = data['features'].shape[1]
    else:
        print('Error: unsupported data format!')
    print(f'Data Loaded: {data_path}, {data["num_features"]} features, {data["num_obervations"]} observations ({np.sum(data["labels"] == 1)} pos, {np.sum(data["labels"] == 0)} neg)')
    return data

def evaluate(label_true, label_predict, dec_values, topN):
    pos_class = 1
    neg_class = 0
    assert len(label_true) == len(label_predict)
    assert len(label_true) == len(dec_values)

    # calculate AUC
    fpr, tpr, _ = roc_curve(label_true, dec_values, pos_label=pos_class)
    auc = roc_auc_score(label_true, dec_values)
    
    results = {
        'auc': auc,
        'roc_X': fpr,
        'roc_Y': tpr
    }
    
    # calculate sensitivity, specificity, and BAC
    tp = np.sum((label_true == pos_class) & (label_predict == pos_class))
    fn = np.sum((label_true == pos_class) & (label_predict == neg_class))
    tn = np.sum((label_true == neg_class) & (label_predict == neg_class))
    fp = np.sum((label_true == neg_class) & (label_predict == pos_class))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    results['bac'] = (sensitivity + specificity) / 2
    results['sensitivity'] = sensitivity
    results['specificity'] = specificity

    # calculate metrics using topN% cut-off thresh
    k = round(len(label_true) * topN)
    idx = np.argsort(dec_values)[::-1]
    label_predict_topk = np.full(len(label_true), neg_class)
    label_predict_topk[idx[:k]] = pos_class
    tp_topk = np.sum((label_true == pos_class) & (label_predict_topk == pos_class))
    fn_topk = np.sum((label_true == pos_class) & (label_predict_topk == neg_class))
    tn_topk = np.sum((label_true == neg_class) & (label_predict_topk == neg_class))
    fp_topk = np.sum((label_true == neg_class) & (label_predict_topk == pos_class))
    sensitivity_topk = tp_topk / (tp_topk + fn_topk)
    specificity_topk = tn_topk / (tn_topk + fp_topk)
    results['bac_topk'] = (sensitivity_topk + specificity_topk) / 2
    precision_topk = tp_topk / (tp_topk + fp_topk)
    results['sensitivity_topk'] = sensitivity_topk
    results['specificity_topk'] = specificity_topk
    results['precision_topk'] = precision_topk

    # calculate NDCG@k
    hits = np.sum(label_true == pos_class)
    kz = min(k, hits)
    z = sum((2**1 - 1) / np.log2(1 + i) for i in range(1, kz + 1))
    dcg_at_k = sum((2**1 - 1) / np.log2(1 + i) if label_true[idx[i]] == pos_class else 0 for i in range(k))
    ndcg_at_k = dcg_at_k / z if z != 0 else 0
    results['ndcg_at_k'] = ndcg_at_k

    return results

# Main loop
with open("results_rusboost_rf.txt", "w") as f:
    # Read training data
    data_train = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', 'data_default', 1991, 2001)
    y_train = data_train['labels']
    X_train = data_train['features']

    # Read testing data
    data_test = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', 'data_default', 2003, 2003)
    y_test = data_test['labels']
    X_test = data_test['features']

    # Train RUSBoost model
    t1 = time.time()
    base_model = DecisionTreeClassifier(min_samples_leaf=5)
    ada_boost = AdaBoostClassifier(base_model, n_estimators=300, learning_rate=0.1)
    ada_boost.fit(X_train, y_train)
    t_train = time.time() - t1

    # Train Random Forest model
    t1_rf = time.time()
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)
    rf.fit(X_train, y_train)
    t_train_rf = time.time() - t1_rf

    # Test RUSBoost model
    t2 = time.time()
    label_predict = ada_boost.predict(X_test)
    dec_values = ada_boost.decision_function(X_test)
    t_test = time.time() - t2

    # Test Random Forest model
    t2_rf = time.time()
    label_predict_rf = rf.predict(X_test)
    dec_values_rf = rf.predict_proba(X_test)[:, 1]
    t_test_rf = time.time() - t2_rf

    # Print performance results for RUSBoost
    print(f'RUSBoost Training time: {t_train:.2f} seconds | Testing time: {t_test:.2f} seconds')

    # Print performance results for Random Forest
    print(f'Random Forest Training time: {t_train_rf:.2f} seconds | Testing time: {t_test_rf:.2f} seconds')

    # Evaluate RUSBoost
    for topN in [0.01, 0.02, 0.03, 0.04, 0.05]:
        metrics = evaluate(y_test, label_predict, dec_values, topN)
        print(f'RUSBoost Performance (top{int(topN*100)}% as cut-off thresh):')
        for key, value in metrics.items():
            print(f'{key}: {value:.4f}')

    # Evaluate RandomForest
    for topN in [0.01, 0.02, 0.03, 0.04, 0.05]:
        metrics_rf = evaluate(y_test, label_predict_rf, dec_values_rf, topN)
        print(f'Random Forest Performance (top{int(topN*100)}% as cut-off thresh):')
        for key, value in metrics_rf.items():
            print(f'{key}: {value:.4f}')
