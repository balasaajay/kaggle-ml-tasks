import numpy as np
import pandas as pd
from pathlib import Path
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import nltk
# Needed only first time to download all nltk libs
# nltk.download('all')

# stop words: Get rid of stop words (a, the, is, ...) since they dont help us in predicting
# Stemming:  to get the root words (Ex: running, run -> run; totally, total -> total) - this will limit the number of words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Read data from the dataset
data = pd.read_csv(Path(f"../dataset/Restaurant_Reviews.tsv"), delimiter='\t', quoting=3)
# dataset is tab seperated
# quoting=3 to ignore double quotes

# print(data.describe())
# print(data.head())

# Initialize stemmer class
ps = PorterStemmer()

# Data cleaning
# Remove stem words and stop words from dataset
corpus = []

# for each review
for i in range(data.shape[0]):
  # get rid of all chars which are not alphabets
  given_review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
  # Apply stemming and remove stopwords
  processed_review = [ps.stem(w) for w in given_review.lower().split() if w not in set(stopwords.words('english'))]
  # print(processed_review)
  corpus.append(' '.join(processed_review))

# print(corpus[0])

# Convert text to numeric format using tf-idf vectorizer
# max_features = number of words to consider for model
# min_df = minimum occurance of word to be considered
# max_df = word occurance should be less than max_df to reduce words that occur frequently.
#          Ex: max_df=0.6 get rids of words that occur in more than 60% of docs
vectorizer = TfidfVectorizer(max_features=150, min_df=3, max_df=0.6)

# Convert corpus to numeric array
X = vectorizer.fit_transform(corpus).toarray()
# print(X)

# Create the output variable
y = data.iloc[:, -1].values
# print(y)

# Split the data to test and train data
# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size = 0.2,
  random_state = 0
)

# Convert data to pytorch tensor format
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)
# print(X_train_tensor.shape, y_train_tensor.shape)
# torch.Size([800, 150]) torch.Size([800])

# Construct Neural network
input_size = 150   # columns from above size
output_size = 2  # positive or negative
hidden_layer_size = 500  # neurons in each NN layer


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
epochs = 300

for epoch in range(epochs):
  optimizer.zero_grad()
  y_train_pred = model(X_train_tensor)
  loss = loss_func(y_train_pred, y_train_tensor)
  loss.backward()
  optimizer.step()
  # print("Epoch: %s; Loss: %s" % (epoch, loss.item()))
  # Epoch: 299; Loss: 0.10705572366714478

print("Model trained!")
# Dictionary and Vectorizer pickle files can be used to load the model and serve as API

# Validate the model
test_input = ['very bad']
test_input_vectorized = vectorizer.transform(test_input).toarray()
test_input_vectorized_tensor = torch.from_numpy(test_input_vectorized).float()
y_test_pred = model(test_input_vectorized_tensor)
_, final_test_preds = torch.max(y_test_pred, -1)
print(final_test_preds)
