.DEFAULT_GOAL := run_titanic_api_container

build_titanic_api_image:
		cp -rf ../titanic/dist/*.whl .
		docker build -t titanic-image:latest .

run_titanic_api_container:
		docker run --rm -d --name titanic_survivor_classifier -p 8000:8000 titanic-image:latest

stop_titanic_api_container:
		docker stop titanic_survivor_classifier

.PHONY: build_titanic_api_image