# Kronos web UI in an isolated container
FROM python:3.11-slim

# CPU-only PyTorch by default (much smaller than the CUDA build)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/kronos/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    KRONOS_HOST=0.0.0.0 \
    KRONOS_PORT=7070 \
    KRONOS_DEBUG=0

WORKDIR /app

COPY requirements.txt /tmp/req/model.txt
COPY webui/requirements.txt /tmp/req/webui.txt
RUN pip install --index-url "$TORCH_INDEX_URL" "torch>=2.1.0" \
    && pip install -r /tmp/req/model.txt -r /tmp/req/webui.txt \
    && rm -rf /tmp/req

# Unprivileged user, code owned by root so it cannot be modified at runtime
RUN useradd --create-home --uid 10001 kronos \
    && mkdir -p /app/data /app/webui/prediction_results /home/kronos/.cache/huggingface \
    && chown kronos:kronos /app/webui/prediction_results /home/kronos/.cache/huggingface

COPY model/ /app/model/
COPY webui/app.py webui/run.py /app/webui/
COPY webui/templates/ /app/webui/templates/

USER kronos
WORKDIR /app/webui
EXPOSE 7070

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7070/api/model-status', timeout=4)" || exit 1

CMD ["python", "app.py"]
