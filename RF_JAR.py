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

def clean_data(X):
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    return X

iters = 3000
gap = 2

for year_test in range(2003, 2009):
    np.random.seed(0)
    print(f'Running RandomForest (training period: 1991-{year_test-gap}, testing period: {year_test}, with {gap}-year gap)...')
    data_train = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', 'uscecchini28', 1991, year_test-gap)
    y_train = data_train['labels']
    X_train = data_train['features']
    newpaaer_train = data_train['newpaaers']
    
    data_test = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', 'uscecchini28', year_test, year_test)
    y_test = data_test['labels']
    X_test = data_test['features']
    newpaaer_test = np.unique(data_test['newpaaers'][data_test['labels'] != 0])

    num_frauds = np.sum(y_train == 1)
    y_train[np.isin(newpaaer_train, newpaaer_test)] = 0
    num_frauds -= np.sum(y_train == 1)
    print(f'Recode {num_frauds} overlapped frauds (i.e., change fraud label from 1 to 0).')

    # Clean data
    X_train = clean_data(X_train)
    X_test = clean_data(X_test)

    t1 = time.time()
    rf = RandomForestClassifier(n_estimators=iters, min_samples_leaf=5, random_state=0)
    rf.fit(X_train, y_train)
    t_train = time.time() - t1

    t2 = time.time()
    label_predict = rf.predict(X_test)
    dec_values = rf.predict_proba(X_test)[:, 1]
    t_test = time.time() - t2

    print(f'Training time: {t_train:.2f} seconds | Testing time: {t_test:.2f} seconds')
    metrics = evaluate(y_test, label_predict, dec_values, 0.01)
    print('Performance (top1% as cut-off thresh):')
    print(f'AUC: {metrics["auc"]:.4f}')
    print(f'NCDG@k=top1%: {metrics["ndcg_at_k"]:.4f}')
    print(f'Sensitivity: {metrics["sensitivity_topk"]*100:.2f}%')
    print(f'Precision: {metrics["precision_topk"]*100:.2f}%')

    output_filename = f'prediction_rf28_{year_test}.csv'
    pd.DataFrame({
        'years': data_test['years'],
        'firms': data_test['firms'],
        'newpaaers': data_test['newpaaers'],
        'y_test': y_test,
        'label_predict': label_predict,
        'dec_values': dec_values
    }).to_csv(output_filename, index=False, float_format='%.6f')
