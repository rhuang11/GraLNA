import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import GridSearchCV, train_test_split

def financial_kernel_transform(X):
    n = X.shape[1]
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            ratios.append(X[:, i] / X[:, j])
            ratios.append(X[:, j] / X[:, i])
    return np.array(ratios).T

def data_reader(data_path, year_start, year_end):
    # Read data from CSV file
    data = pd.read_csv(data_path)

    # Filter data based on years
    data = data[(data['fyear'] >= year_start) & (data['fyear'] <= year_end)]

    # Separate features and labels
    X = data.iloc[:, 4:].values  # Assuming features start from column index 4
    y = data['misstate'].values

    return X, y

# Set parameters
start_year = 1991
end_year = 2014

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Year_Test', 'AUC', 'NDCG@1%', 'NDCG@2%', 'NDCG@3%', 'NDCG@4%', 'NDCG@5%'])

# Loop through each testing year
for year_test in range(2003, 2015):
    print(f"==> Running SVM-FK (training period: {start_year}-{year_test-2}, testing period: {year_test}, with 2-year gap)...")
    
    # Read training data
    X_train_raw, y_train = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', start_year, year_test-2)

    # Read testing data
    X_test_raw, y_test = data_reader('~/GraLNA/data_FraudDetection_JAR2020.csv', year_test, year_test)

    # Apply financial kernel transformation
    X_train = financial_kernel_transform(X_train_raw)
    X_test = financial_kernel_transform(X_test_raw)

    # Use a holdout set for parameter tuning
    X_train_tune, X_val, y_train_tune, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Define the SVM model with class weights for cost-sensitive learning
    param_grid = {'C': [0.1, 1, 10, 20, 50, 100]}
    svc = SVC(kernel='linear', class_weight='balanced', probability=True)
    clf = GridSearchCV(svc, param_grid, scoring='roc_auc', cv=5)

    # Train the model and tune parameters
    clf.fit(X_train_tune, y_train_tune)
    best_model = clf.best_estimator_

    # Train the best model on the entire training set
    best_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate NDCG@k
    k_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    ndcg_scores = []
    for k in k_values:
        k_top = int(len(y_test) * k)
        ndcg = ndcg_score([y_test], [y_pred_proba], k=k_top)
        ndcg_scores.append(ndcg)

    # Store results in DataFrame
    results_df = results_df.append({'Year_Test': year_test,
                                    'AUC': auc,
                                    'NDCG@1%': ndcg_scores[0],
                                    'NDCG@2%': ndcg_scores[1],
                                    'NDCG@3%': ndcg_scores[2],
                                    'NDCG@4%': ndcg_scores[3],
                                    'NDCG@5%': ndcg_scores[4]}, ignore_index=True)

# Write results to CSV
results_df.to_csv('results_svm_fk.csv', index=False)
