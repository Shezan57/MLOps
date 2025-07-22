import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

clf = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv("./data/features/test_bow.csv")

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Create a dictionary to hold the evaluation results
evaluation_results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

# Save the evaluation results to a JSON file
with open('metrics.json', 'w') as f:
    json.dump(evaluation_results, f, indent=4)

