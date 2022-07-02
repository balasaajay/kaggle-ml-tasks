# NLP Classifier 

Check resturant reviews and determine positive or negative review.

### Dataset
Dataset from Kaggle: https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews

### Classifier Used
K-NN classifier is used to build the model

### Vectorizer Used
Vectorizer used is TF-IDF (term frequency-inverse document frequency). 
Picked Tf-idf since BagofWords doesnt give weights to the words.

### Build/Train the model
        python text_classifier_knn.py

### Validate the model
        python -m pytest validation.py 
