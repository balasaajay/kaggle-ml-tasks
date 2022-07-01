FROM python:3.9.4

# Create the user that will run the app
RUN adduser --disabled-password --gecos '' api-user

WORKDIR /opt/ml_api

# Install requirements, including from Gemfury
ADD ./ml_api /opt/ml_api
RUN pip install --upgrade pip
RUN pip install -r /opt/ml_api/requirements.txt

COPY *.whl /opt/ml_api/

RUN pip install /opt/ml_api/*.whl

ENV PYTHONPATH $PYTHONPATH:/opt

COPY run.sh /opt/ml_api/
RUN chmod a+rx /opt/ml_api/run.sh
RUN chown -R api-user:api-user ./

USER api-user

EXPOSE 8000

CMD ["bash", "./run.sh"]
