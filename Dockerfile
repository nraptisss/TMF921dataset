FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY seeds ./seeds
COPY tr290-docs ./tr290-docs
COPY resources ./resources
COPY 921-v5.0.0-ctk-1.0.0 ./921-v5.0.0-ctk-1.0.0
COPY tmf921-v5.0.0-ri-1.0.0 ./tmf921-v5.0.0-ri-1.0.0

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

CMD ["python", "-m", "tmf921_dataset_gen.cli", "generate", "--count", "20"]
