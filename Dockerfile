# Base image with Python
FROM python:3.12-slim

# Install build tools and useful Debian packages
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    tree \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src ./src

# BUGFIX: overwrite langgraph config file
COPY ./bugfix/config.py /usr/local/lib/python3.12/site-packages/langgraph/utils/config.py
