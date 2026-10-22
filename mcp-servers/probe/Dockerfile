# Measurement server, not part of the deployed agent — see server.js for what it answers.
# Node base plus one file, deliberately: a container with no startup work of its own makes
# the platform's own snapshot and boot cost the only thing being measured.
FROM public.ecr.aws/docker/library/node:24-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.js /app/server.js

EXPOSE 8000

# Non-root: it holds no credential and writes only its own marker under /tmp.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser
ENV HOME=/home/appuser

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/ || exit 1

CMD ["node", "/app/server.js"]
