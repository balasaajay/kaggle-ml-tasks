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
- Given a review, this NLP model based on K-NN predicts whether it is positive review or not.
- Kaggel link - https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews
- (Tests developed using pytest framework)
