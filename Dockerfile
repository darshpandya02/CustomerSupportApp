# Container build for running the API outside Vercel (for example on a VM or Kubernetes).
# The frontend in public/ is served by the same process when VERCEL is unset.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app.py ./
COPY supportbot ./supportbot
COPY artifacts ./artifacts
COPY public ./public
COPY reports/*.json ./reports/

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
# Needs DATABASE_URL, AGENT_PASSWORD, SESSION_SECRET and AI_GATEWAY_API_KEY at runtime.
CMD ["uv", "run", "--no-sync", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
