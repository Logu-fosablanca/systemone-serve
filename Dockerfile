FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=""
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV DEVICE="cuda"
ENV DTYPE="bfloat16"
ENV MAX_CONCURRENT="4"
ENV NOUL_YES_TOKEN="yes"
ENV NOUL_NO_TOKEN="no"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
