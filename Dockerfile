FROM python:3.12-slim

ARG PUID=3000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    DATA_DIR=/data \
    JOBS_DIR=/jobs \
    SNAPSHOT_DIR=/snapshots

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid ${PUID} --gid 0 --create-home --home-dir /home/ollma ollma

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY backend /app/backend
COPY public /app/public

USER ${PUID}:0
EXPOSE 8080
CMD ["python", "/app/app.py"]
