import warnings
from sklearnex import patch_sklearn
patch_sklearn()
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('~/GraLNA/data_FraudDetection_JAR2020.csv')
df.head()

target = df.misstate
print(target)

df = df.drop(['fyear', 'gvkey', 'p_aaer', 'misstate'], axis=1)
df.head()

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
scaled_df.head()

X_train, X_test, y_train, y_test = train_test_split(scaled_df, target, test_size=0.25)

print("Handling missing values in training data...")
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)

adaboost_clf = AdaBoostClassifier()
gbt_clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=10,
                                    max_features=0.35000000000000003, min_samples_leaf=10, min_samples_split=5,
                                     n_estimators=100, subsample=0.7500000000000001)

adaboost_clf.fit(X_train, y_train)

gbt_clf.fit(X_train, y_train)

#gbt_clf.fit(np.asmatrix(X_train), np.asmatrix(y_train))

adaboost_train_preds = adaboost_clf.predict(X_train)
adaboost_test_preds = adaboost_clf.predict(X_test)
gbt_clf_train_preds = gbt_clf.predict(X_train)
gbt_clf_test_preds = gbt_clf.predict(X_test)

def display_acc_and_f1_score(true, preds, model_name):
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)
    print("Model: {}".format(model_name))
    print("Accuracy: {}".format(acc))
    print("F1-Score: {}".format(f1))

print("Training Metrics")
display_acc_and_f1_score(y_train, adaboost_train_preds, model_name='AdaBoost')
print("")
display_acc_and_f1_score(y_train, gbt_clf_train_preds, model_name='Gradient Boosted Trees')
print("")
print("Testing Metrics")
display_acc_and_f1_score(y_test, adaboost_test_preds, model_name='AdaBoost')
print("")
display_acc_and_f1_score(y_test, gbt_clf_test_preds, model_name='Gradient Boosted Trees')

adaboost_confusion_matrix = confusion_matrix(y_test, adaboost_test_preds)
adaboost_confusion_matrix

gbt_confusion_matrix = confusion_matrix(y_test, gbt_clf_test_preds)
gbt_confusion_matrix

adaboost_classification_report = classification_report(y_test, adaboost_test_preds)
print(adaboost_classification_report)

gbt_classification_report = classification_report(y_test, gbt_clf_test_preds)
print(gbt_classification_report)

print('Mean Adaboost Cross-Val Score (k=5):')
print(cross_val_score(adaboost_clf, scaled_df, target, cv=5).mean())

print('Mean GBT Cross-Val Score (k=5):')
print(cross_val_score(gbt_clf, scaled_df, target, cv=5).mean())
