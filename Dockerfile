FROM node:20-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WELL_SEISMIC_HOST=0.0.0.0 \
    WELL_SEISMIC_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
COPY --from=frontend /build/frontend/dist /app/frontend/dist
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir ".[all]" \
    && python -m pip install --no-cache-dir -e "./接口模型/cigvis-main/cigvis-main[viser]" \
    && python tools/verify_release.py --runtime

EXPOSE 8000 8080
CMD ["python", "-m", "well_seismic.api"]

