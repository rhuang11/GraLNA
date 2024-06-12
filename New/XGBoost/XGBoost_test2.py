import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

finfraud_copy = pd.read_csv('~/GraLNA/New/finfraud_copy.csv')

# Initialize the results DataFrame
results = pd.DataFrame(columns=['year', 'auc', 'accuracy', 'precision', 'recall', 'true_positives', 'false_positives', 'false_negatives'])

# Assume 'finfraud_copy' is your DataFrame
X1 = finfraud_copy.drop(['misstate', 'p_aaer', 'gvkey'], axis=1)
y1 = finfraud_copy['misstate']

# Normalize the values in X1
scaler = StandardScaler()
X1_normalized = scaler.fit_transform(X1)

# Adding the normalized features back into the DataFrame for easy indexing
X1_normalized_df = pd.DataFrame(X1_normalized, columns=X1.columns)

# Training data is fyear = 1991-2001, testing data is fyear 2003 initially, then fyear 2004 but also expand training data to go up one year every time as well
for year in range(2003, 2009):
    X_train = X1_normalized_df[X1['fyear'] <= year - 2]
    X_test = X1_normalized_df[X1['fyear'] == year]
    y_train = y1[X1['fyear'] <= year - 2]
    y_test = y1[X1['fyear'] == year]

    # Calculate the class weight ratio
    class_weight_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Create an XGBoost classifier with class weights
    model = XGBClassifier(scale_pos_weight=class_weight_ratio, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_proba)
    print("AUC for year {}: {}".format(year, auc))

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for year {}: {}".format(year, accuracy))

    # Calculate precision and recall
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision for year {}: {}".format(year, precision))
    print("Recall for year {}: {}".format(year, recall))

    # Calculate true positives, false positives, and false negatives
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("True Positives for year {}: {}".format(year, tp))
    print("False Positives for year {}: {}".format(year, fp))
    print("False Negatives for year {}: {}".format(year, fn))

    # Add results to DataFrame
    results = results.append({'year': year, 'auc': auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'true_positives': tp, 'false_positives': fp, 'false_negatives': fn}, ignore_index=True)

print(results)

results.to_csv('/Users/ryanhuang/Developer/GraLNA/New/XGBoost_results.csv', index=False)
