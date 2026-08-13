FROM python:3.11-slim

WORKDIR /app
COPY *.py /app/

ENV PORT=8080 \
    LOAD_RPS=200 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python", "recommendation_server.py"]
