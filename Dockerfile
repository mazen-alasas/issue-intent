FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ISSUE_INTENT_BACKEND=onnx-int8 \
    ISSUE_INTENT_CHECKPOINT=/models/distilbert \
    ISSUE_INTENT_ONNX_MODEL=/models/onnx/model_int8.onnx

WORKDIR /app

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /app/src/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
