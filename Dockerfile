FROM python:3.8

RUN mkdir -p /app
WORKDIR /app

# todo: remove
RUN git clone -b enrollments-caching https://github.com/scholtzan/mozanalysis.git
RUN pip install mozanalysis

COPY . .
RUN pip install -r requirements.txt

RUN pip install .

ENTRYPOINT ["jetstream"]
