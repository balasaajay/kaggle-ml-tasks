#!/bin/bash

curl -X POST \
  http://localhost:8501/v1/models/customer_sentiment_model:predict \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 4d69ba39-af84-d0cb-1680-8b30949163f2' \
  -d '{
	"instances": [[-1.433,-0.47]],
	"signature_name": "serving_default"
}'