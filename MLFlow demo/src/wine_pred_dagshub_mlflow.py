import dagshub
import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='Shezan57', repo_name='dagshub_experiment', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Shezan57/dagshub_experiment.mlflow")

mlflow.set_experiment("Wine_Classification_Experiment_Dagshub")


 # Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():

    # Define model parameters
    max_depth = 5
    n_estimators = 100

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators",n_estimators )
    
    # create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # save plot
    plt.savefig("confusion_matrix.png")
    
    # Log the confusion matrix as an artifact
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    
    # tags
    mlflow.set_tags({"Author": 'Shezan', "Project": "Wine Classification"})
    
    # log the model
    mlflow.sklearn.log_model(rf, "RandomForest_Model")
    
    print(accuracy)
    
    
    
    
    