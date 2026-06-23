# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv
RUN uv sync --frozen

RUN uv run python -m spacy download en_core_web_lg

# Copy the application code
COPY ./app /code/app
COPY main.py /code/
COPY model.py /code/
COPY data_loader.py /code/
COPY trainer.py /code/
COPY evaluator.py /code/
COPY checkpoints.py /code/
COPY utils.py /code/
COPY generator.py /code/
COPY ./checkpoints /code/checkpoints

# Command to run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]