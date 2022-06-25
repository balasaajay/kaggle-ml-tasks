# Kaggle Datasets
Contains code to build, train and package a model.
Model developed and structured based on learnings from https://www.udemy.com/course/deployment-of-machine-learning-models/

## Commands:
cd to a model directory and run below commands

### Requirements:
  - Install tox using pip
        pip install tox

### Install dependencies and to train a model:
        tox -e train

### Run linter
        tox -e lint

## Titanic
Used - Logistic regression to build the model.
URL: https://www.kaggle.com/datasets/heptapod/titanic
Dataset: https://www.openml.org/data/get_csv/16826755/phpMYEkMl


# ML Pipelines Overview:

Data (reproducible -> R) --> Data Analysis --> Data Pre-processing(R) --> Variable Selection(R) --> ML model building(R) --> Model deploy(R)
                                               (Feature Engineering)       (Feature Selection)

Data Layer --> Feature layer --> Scoring layer --> Evaluation layer 
(base)
