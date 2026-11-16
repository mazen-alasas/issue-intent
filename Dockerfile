FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ISSUE_INTENT_CHECKPOINT=/models/distilbert

WORKDIR /app

# Install CPU-only PyTorch.
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install API runtime dependencies.
COPY docker/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir \
    -r /tmp/requirements.txt

# Copy application source only.
COPY src/ /app/src/

# Run the API as a non-root user.
RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
