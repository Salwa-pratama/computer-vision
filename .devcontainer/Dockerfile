FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Container
CMD ["sleep","infinity"]

