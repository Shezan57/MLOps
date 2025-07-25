import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/preprocessed
train_data = pd.read_csv("./data/processed/train_processed_data.csv")
test_data = pd.read_csv("./data/processed/test_processed_data.csv")

train_data.fillna("", inplace=True)
test_data.fillna("", inplace=True)

# Apply BoW
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=50)

# Fit and transform the training data
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Save the processed data inside the data/features
data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train_bow.csv"))
test_df.to_csv(os.path.join(data_path, "test_bow.csv"))