FROM python:3.8-alpine

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt

RUN apk add --no-cache --virtual .build-deps gcc musl-dev && \
    pip install -r requirements.txt && \
    apk del .build-deps

COPY . .

RUN python setup.py install

ENTRYPOINT ["/app/bin/entrypoint"]
