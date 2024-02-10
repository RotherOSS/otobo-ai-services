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

RUN pip install poetry==1.6.1

#TODO: Remove test sr
RUN pip install elasticsearch

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./mylibs ./mylibs

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi
