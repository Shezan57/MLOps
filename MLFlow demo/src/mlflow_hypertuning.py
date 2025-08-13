import dagshub
import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import pandas as pd

dagshub.init(repo_owner='Shezan57', repo_name='dagshub_experiment', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Shezan57/dagshub_experiment.mlflow")

mlflow.set_experiment("Wine_Classification_Experiment_Dagshub_Hyperparameter_Tuning")

# Load Dataset
wine = load_wine()
X = wine.data
y = wine.target

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
rf = RandomForestClassifier()

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'criterion': ["gini", "entropy", "log_loss"]
}

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

with mlflow.start_run() as parent:

    
    grid_search.fit(X_train, y_train)

    # Log each parameter combination
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])
    
    # Display the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Log best parameters 
    mlflow.log_params(best_params)
    
    # log the metrics
    mlflow.log_metric("best_accuracy", best_score)
    
    #Log training data
    train_df = pd.DataFrame(X_train, columns=wine.feature_names)
    train_df["target"] = y_train
    
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training_data")

    test_df = pd.DataFrame(X_test, columns=wine.feature_names)
    test_df["target"] = y_test
    
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing_data")

    # Evaluate model
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save plot
    plt.savefig("confusion_matrix_grid_search.png")

    # Log the confusion matrix as an artifact
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({"Author": 'Shezan', "Project": "Wine Classification Grid Search"})

    # Log the best model
    mlflow.sklearn.log_model(best_rf, "RandomForest_Model")

    print("accuracy: ", accuracy)
    print("best params", best_params)
