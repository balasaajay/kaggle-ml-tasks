# ML Flow
MLFlow is used to build ML Pipelines and track metrics

### To run the model:
        python knn_model_mlflow.py

### To start the MLFlow UI:
        mlflow ui
Navigate to http://127.0.0.1:5000

### To deploy a model using MLFlow
        mlflow models serve --model-uri runs:/<run-id>/model --port 1244
App will be available at http://127.0.0.1:1244/invocations
