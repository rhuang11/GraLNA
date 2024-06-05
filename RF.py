import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

# Load the dataset
file_path = "~/GraLNA/data_FraudDetection_JAR2020.csv"
df = pd.read_csv(file_path)

# Prepare the data
X = df.drop(columns=['misstate', 'fyear', 'p_aaer', 'gvkey'])
y = df['misstate']

X = X.fillna(df.median())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sample non-fraudulent instances equal to the number of fraudulent ones
fraud_indices = np.where(y_train == 1)[0]
non_fraud_indices = np.where(y_train == 0)[0]
num_frauds = len(fraud_indices)
sampled_non_fraud_indices = np.random.choice(non_fraud_indices, size=num_frauds, replace=False)
selected_indices = np.concatenate((fraud_indices, sampled_non_fraud_indices))

X_train_sampled = X_train.iloc[selected_indices]
y_train_sampled = y_train.iloc[selected_indices]

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'max_features': [3, 5, 7]
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_sampled, y_train_sampled)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test data
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)

# Print Results
print("Best Parameters:", best_params)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")

# Train the best model on the entire dataset
best_model.fit(X, y)

# Make predictions on the entire dataset
df['predictions'] = best_model.predict(X)

# Save predictions to a file
df.to_csv("predictions.csv", index=False)
