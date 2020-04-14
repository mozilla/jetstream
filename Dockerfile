FROM python:3.8-slim

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN pip install -U --extra-index-url https://pypi.fury.io/arrow-nightlies/ --pre pyarrow

COPY . .

ENTRYPOINT ["/app/bin/entrypoint"]
