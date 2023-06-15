FROM python:3.10.11-bullseye

workdir /dermaflow

COPY dermaflow .
copy model model/
COPY packages.txt requirements.txt
COPY setup.py setup.py
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .
CMD uvicorn api.backend.api:app --host 0.0.0.0 --port $PORT
