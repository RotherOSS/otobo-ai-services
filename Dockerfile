# This is the build file for the OTOBO AI docker image.

FROM python:3.12-slim

# Install required build tools and optional Debian packages
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    build-essential \
    g++ \
    less \
    nano \
    rsync \
    telnet \
    tree \
    screen \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.4

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./
COPY ./src ./src

RUN poetry lock --no-update
RUN poetry install --no-interaction --no-ansi

### BUGFIX
COPY ./bugfix/config.py /usr/local/lib/python3.12/site-packages/langgraph/utils/config.py

# EXPOSE 8080

CMD exec uvicorn src.server:app --host 0.0.0.0 --port 8080 --loop asyncio
