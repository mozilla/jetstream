FROM python:3.8-slim

RUN mkdir -p /app
WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN apt-get install -y gcc

COPY . .
RUN pip install -r requirements.txt

RUN pip install .

ENTRYPOINT ["/app/bin/entrypoint"]
