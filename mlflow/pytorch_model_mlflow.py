from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from pathlib import Path

import mlflow
import mlflow.pytorch


def main():
    mlflow.set_experiment(experiment_name="mlflow pytorch")
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

    # Convert data to pytorch tensor format
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).long()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    # print(X_train_tensor.shape, y_train_tensor.shape)

    # Construct Neural network
    input_size = 2   # age, salary
    output_size = 2  # yes or no
    hidden_layer_size = 10  # neurons in each NN layer

    class Net(nn.Module):
      def __init__(self):
        super().__init__()
        # Build 3 fully connected layers in which 2 are hidden
        self.fcl1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fcl2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fcl3 = torch.nn.Linear(hidden_layer_size, output_size)
      
      def forward(self, X):
        # Use relu actiivation function for hidden layers
        X= torch.relu((self.fcl1(X)))
        X = torch.relu((self.fcl2(X)))
        X = self.fcl3(X)
        # Use softmax activation for output layer
        return F.log_softmax(X, dim=1)

    # Create a model from Net
    model = Net()

    # Optimization, Learning rate and Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    loss_func = nn.NLLLoss()

    # Number of times NN has to run end-end to train and adjust weights
    epochs = 200

    for epoch in range(epochs):
      optimizer.zero_grad()
      y_train_pred = model(X_train_tensor)
      loss = loss_func(y_train_pred, y_train_tensor)
      loss.backward()
      optimizer.step()
      print("Epoch: %s; Loss: %s" % (epoch, loss.item()))
      # Epoch: 99; Loss: 0.1401793360710144

    mlflow.end_run()
    with mlflow.start_run() as run:
      mlflow.log_param("epochs", 100)
      mlflow.pytorch.log_model(model, "models")
      mlflow.log_metric("loss", loss.item())

    y_test_pred = model(torch.from_numpy(sc.transform(np.array([[43, 60000]]))).float())
    # print(y_test_pred)
    _, final_test_preds = torch.max(y_test_pred, -1)
    # print(final_test_preds)

if __name__ == "__main__":
    main()