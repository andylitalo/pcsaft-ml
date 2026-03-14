# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies with pip
RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code and model artifacts
COPY model/ model/
COPY serving/ serving/
COPY screening/ screening/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
