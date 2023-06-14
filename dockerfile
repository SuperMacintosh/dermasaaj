FROM python:3.10.11-bullseye
COPY dermaflow /dermaflow
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn dermaflow.api.backend.api:app --host 0.0.0.0 --port $PORT
