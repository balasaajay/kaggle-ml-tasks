## Fast API to serve the model as API
API to serve wine model. 
Dataset details: https://archive.ics.uci.edu/ml/datasets/wine

### Using fastAPI and uvicorn to run the ml model api
        cd wine_dataset
        pip install -r requirements.txt
        uvicorn main:app      # start the server

#### Open the browser and navigate to http://127.0.0.1:8000/docs for accessing the APIs
