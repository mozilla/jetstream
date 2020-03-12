FROM continuumio/miniconda3

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY constraints.txt constraints.txt

RUN conda install scipy==1.3.1
RUN pip install -c constraints.txt -r requirements.txt

COPY . .

ENTRYPOINT ["/app/bin/entrypoint"]
