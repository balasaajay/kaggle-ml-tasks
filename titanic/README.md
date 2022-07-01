# Kaggle Datasets
Contains code to build, train and package a model.
Model developed and structured based on learnings from https://www.udemy.com/course/deployment-of-machine-learning-models/

## Commands:
cd to a model directory and run below commands

### Requirements:
        pip install tox

### Install dependencies and to train a model:
        tox -e train

### Run linter
        tox -e lint

### To train, run linter and stylechecks
        tox

### To build .whl package
        python -m build

### Using fastAPI and uvicorn to run the ml model api
        cd api-app/ml_api
        pip install -r requirements.txt
        pip install titanic_logres_model-0.0.1-py3-none-any.whl   # Install Titanic model wheel package
        uvicorn main:app      # start the server

#### Open the browser and navigate to http://127.0.0.1:8000/docs for accessing the APIs

### To build docker image with titanic api
        cd api-app
        make build_titanic_api_image

### To run container to serve titanic api (Default make task)
        cd api-app
        make run_titanic_api_container

### To stop titanic container
        cd api-app
        make stop_titanic_api_container

## Titanic
- Logistic regression to build the model.
- URL: https://www.kaggle.com/datasets/heptapod/titanic
- Dataset: https://www.openml.org/data/get_csv/16826755/phpMYEkMl
