# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir flask waitress
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 80

STOPSIGNAL SIGINT

ENTRYPOINT ["python3"]
CMD ["server.py"]
