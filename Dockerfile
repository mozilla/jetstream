FROM python:3.8

RUN mkdir -p /app
WORKDIR /app

COPY . .
RUN pip install -r requirements.txt

RUN pip install .

ENTRYPOINT ["/app/bin/entrypoint"]
