FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==3.11.1 \
    psycopg2-binary==2.9.11 \
    boto3==1.36.0

EXPOSE 5000
