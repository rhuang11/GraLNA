import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
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

def clean_data(X):
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    return X

# Read training data
data_train = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', 'uscecchini28', 1991, 2001)
y_train = data_train['labels']
X_train = data_train['features']
newpaaer_train = data_train['newpaaers']

# Read testing data
data_test = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', 'uscecchini28', 2003, 2003)
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

# Identify the indices of fraudulent and non-fraudulent samples
fraud_indices = np.where(y_train == 1)[0]
non_fraud_indices = np.where(y_train == 0)[0]

# Randomly sample the same number of non-fraudulent indices as fraudulent ones
sampled_non_fraud_indices = np.random.choice(non_fraud_indices, size=len(fraud_indices), replace=False)

# Combine fraudulent and sampled non-fraudulent indices
selected_indices = np.concatenate((fraud_indices, sampled_non_fraud_indices))

# Use the selected indices to subset the features and labels for training
X_train_sampled = X_train[selected_indices]
y_train_sampled = y_train[selected_indices]

# Train RandomForestClassifier using the sampled data
t1 = time.time()
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)
rf.fit(X_train_sampled, y_train_sampled)
t_train = time.time() - t1

t2 = time.time()
label_predict = rf.predict(X_test)
dec_values = rf.predict_proba(X_test)[:, 1]
t_test = time.time() - t2

accuracy = accuracy_score(y_test, label_predict)
print(f'Accuracy: {accuracy}')

# Create a DataFrame with the results
results_df = pd.DataFrame({'Actual Labels': y_test, 'Predicted Labels': label_predict, 'Decision Values': dec_values})

# Save the DataFrame to a CSV file
results_df.to_csv('results.csv', index=False)

