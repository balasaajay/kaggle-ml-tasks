# ML Models
Contains code to build, train and package a model.
Model developed and structured based on learnings from https://www.udemy.com/course/deployment-of-machine-learning-models/

# ML Pipelines Overview:

Data (reproducible -> R) --> Data Analysis --> Data Pre-processing(R) (Feature Engineering) --> Variable Selection(R)(Feature Selection) --> ML model building(R) --> Model deploy(R)

Data Layer(base) --> Feature layer --> Scoring layer --> Evaluation layer 


# Models in this repo:

### Titanic
- Classifier model based on Logistic regression to predict survivors in Titanic
- Kaggle link - https://www.kaggle.com/datasets/heptapod/titanic

### Customer Behavior
- Given customer details, this tensorflow-keras/pytorch neural network (with 2 hidden layers) based model predicts the probability of the customer likeliness to buy or not

### Restaurant reviews
- Given a review, this NLP model based on K-NN/tensorflow-keras/pytorch neural network predicts whether it is positive review or not.
- Kaggel link - https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews
- (Tests developed using pytest framework)


# Others in this repo:

### Tensorflow JS
- Develop ML models in JavaScript, and use ML directly in the browser or in Node.js
- This repo has a html file which uses tfjs to run a model in the client browser
- https://www.tensorflow.org/js


### MLFlow
- This repo has KNN(sklearn) and Pytorch models integrated with mlflow.
- Opensource app for ML lifecycle management
- MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry. 
- https://mlflow.org/