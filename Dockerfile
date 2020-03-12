FROM python:3.8-slim

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["/app/bin/entrypoint"]
