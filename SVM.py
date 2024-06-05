import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

# Load the dataset
file_path = "/Users/ryanhuang/Downloads/data_FraudDetection_JAR2020.csv"
df = pd.read_csv(file_path)

# Fill missing values with median
df = df.fillna(df.median())

# Prepare the data
X = df.drop(columns=['misstate', 'fyear', 'p_aaer', 'gvkey'])
y = df['misstate']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Evaluate on Test Data
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)

# Print Results
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
