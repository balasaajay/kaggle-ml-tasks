import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from pathlib import Path
import pickle

# Load the data
data = pd.read_csv(Path(f"../dataset/storepurchasedata_large.csv"))
# print(data.describe())

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(
  data.iloc[:, :-1].values,
  data.iloc[:, -1].values,
  test_size = 0.8,
  random_state = 0
)
# print(X_train, X_test)

# Feature scale the train and test data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build model using tf.keras models Sequential class
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax') # Output size = 2
])

# set optimizer and loss functions
model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs = 50)

# Collect model metrics
loss, accuracy = model.evaluate(X_test, y_test)
# print(accuracy)
# print(loss)

test_prediction = model.predict(sc.transform(np.array([[42, 50000]])))[:, 1]
# print(test_prediction)

# Save and server hte model using tf serving
model.save('customer_sentiment_model/1') # saves model in protobuf format
# variables stores the weights of the models

# Save scaler in pickle file
scaler_file = "sc.pickle"
pickle.dump(sc, open(scaler_file, 'wb'))
