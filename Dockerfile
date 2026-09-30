# Multi-Agent Research Assistant — API + worker image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY pyproject.toml .

# Non-root runtime user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: run the FastAPI service. Override command for the worker:
#   docker run ... <image> python -m app.workers.runner
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
