# This is the build file for the OTOBO AI docker image.

FROM python:3.11-slim

# Install some required and optional Debian packages.
RUN apt-get update\
    && apt-get -y --no-install-recommends install\
    "less"\
    "nano"\
    "rsync"\
    "telnet"\
    "tree"\
    "screen"\
    "vim"\
    "curl"\
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.4

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./mylibs ./mylibs

RUN poetry lock --no-update
RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry lock --no-update
RUN poetry install --no-interaction --no-ansi

### Ist das in der docker_compose yml?
# EXPOSE 8080

# CMD exec uvicorn app.server:app --host 0.0.0.0 --port 8080 --loop asyncio
