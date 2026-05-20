# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for Cloud Run security best practices
RUN useradd --create-home --shell /bin/bash chipllm
USER chipllm
WORKDIR /home/chipllm/app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --chown=chipllm:chipllm . .

# Cloud Run injects $PORT (default 8080)
ENV PORT=8080

EXPOSE 8080

# Streamlit reads PORT from config.toml (set to 8080) — matches Cloud Run
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
