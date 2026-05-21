# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for Cloud Run security best practices
RUN useradd --create-home --shell /bin/bash chipllm
USER chipllm
WORKDIR /home/chipllm/app

# Copy the self-contained virtual environment and update PATH
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=chipllm:chipllm . .

EXPOSE 8080

# Run via shell invocation so Streamlit dynamically binds to Cloud Run's $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]