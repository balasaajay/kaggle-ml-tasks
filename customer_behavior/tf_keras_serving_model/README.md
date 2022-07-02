# Model implementation using Tensorflow and keras

## Intro:
NN Model in this section is implemented using tensorflow and Keras. Model is saved using tf save and is served using tensorflow serving API in a container


## Build and deploy commands

### To build and save the model
        python tf_keras_model.py

This will generate the pb file of model, trained weights.

### To start model as API
        docker-compose up -d

### To stop the container
        docker-compose down

### To validate running service using curl
        ./validate_api.sh
