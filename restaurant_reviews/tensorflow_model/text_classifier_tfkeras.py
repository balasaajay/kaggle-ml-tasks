import numpy as np
import pandas as pd
from pathlib import Path
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf

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

# Construct Neural network
input_size = 150   # columns from above size
output_size = 2  # positive or negative
hidden_layer_size = 500  # neurons in each NN layer

# Build model using tf.keras models Sequential class
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
  tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
  tf.keras.layers.Dense(output_size, activation='softmax') # Output size = 2
])

# set optimizer and loss functions
model.compile(optimizer='adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs = 200)

# Collect model metrics
loss, accuracy = model.evaluate(X_test, y_test)

print("Model trained!")
# Dictionary and Vectorizer pickle files can be used to load the model and serve as API

# Save and server hte model using tf serving
model.save('restaurant_reviews_model/1') # saves model in protobuf format
# variables stores the weights of the models

# Save vectorizer in pickle file
vectorizer_file = "tf-idf-vector-tfk.pickle"
pickle.dump(vectorizer, open(vectorizer_file, 'wb'))

# Validate the model
test_input = ['very bad']
test_input_vectorized = vectorizer.transform(test_input).toarray()
y_test_pred = model(test_input_vectorized)[:, 1]
print(y_test_pred)
# very low number = negative sentiment
