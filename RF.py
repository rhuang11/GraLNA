import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load the dataset
file_path = "~/GraLNA/data_FraudDetection_JAR2020.csv"
df = pd.read_csv(file_path)

# Handle missing values
X = df.drop(columns=['misstate', 'fyear', 'p_aaer', 'gvkey'])
y = df['misstate']

# Impute missing values using median strategy
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, stratify=y, random_state=42)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

# Define the pipeline
pipeline = Pipeline([('smote', smote), ('rf', rf_model)])

# Define the grid of hyperparameters to search
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 5, 10, 15],
    'rf__max_features': [3, 5, 7]
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test data
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Print Results
print("Best Parameters:", best_params)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1_score:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Train the best model on the entire dataset
best_model.fit(X_imputed, y)

# Make predictions on the entire dataset
df['predictions'] = best_model.predict(X_imputed)

# Calculate accuracy on the entire dataset
full_dataset_accuracy = accuracy_score(y, df['predictions'])

# Print accuracy on the entire dataset
print(f"Accuracy on the entire dataset: {full_dataset_accuracy:.4f}")

# Save predictions to a file
df.to_csv("predictions.csv", index=False)
