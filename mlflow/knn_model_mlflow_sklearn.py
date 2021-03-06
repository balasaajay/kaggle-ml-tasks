import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

import mlflow
import mlflow.sklearn

def main():

    mlflow.set_experiment(experiment_name="mlflow_test")
    mlflow.set_experiment_tags({'type':'dev'})
    # Load the data
    data = pd.read_csv(Path(f"dataset/storepurchasedata_large.csv"))
    # print(data.describe())
    mlflow.log_param("Data shape", data.shape)
    # Split the data for train and test
    X_train, X_test, y_train, y_test = train_test_split(
      data.iloc[:, :-1].values,
      data.iloc[:, -1].values,
      test_size = 0.2,
      random_state = 0
    )
    # print(X_train, X_test)

    # Feature scale the train and test data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print("Build and train the model")
    # Classification Model using K-Nearest Neighbors (K=5)
    mlflow.log_param("n_neighbors", 5)
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    model.fit(X_train, y_train)
    print("Model trained")

    # Predict with trained classifier
    y_pred = model.predict(X_test)
    # Probability values of each predicition
    y_probability = model.predict_proba(X_test)[:, 1]
    # print(y_probability)

    # Model metrics:
    # 1. Using Confusion Matrix 
    # (Displays - True Negative, False Positive, False Negative, True Porsitive)
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    # 2. Model Accuracy:
    model_accuracy = accuracy_score(y_test, y_pred)
    print(model_accuracy)

    # Track model accuracy and model
    mlflow.log_metric("accuracy", model_accuracy)
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()