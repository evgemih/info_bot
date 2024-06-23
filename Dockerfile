FROM python:3.11-slim as base

ARG TG_BOT_TOKEN
ENV TG_BOT_TOKEN=$TG_BOT_TOKEN

RUN apt-get update -y \
    && rm -rf /var/lib/apt/lists/*

ENV ROOT /app
WORKDIR $ROOT
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "app.py" ]
