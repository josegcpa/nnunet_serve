FROM python:3.11.4-slim-bullseye
USER root
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=50422

# install internal dependencies
COPY docker-installations.sh docker-installations.sh
RUN bash docker-installations.sh

# install environment
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable
    
# Sync the project
COPY src src
RUN chown root -R src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

COPY model-serve-spec-docker.yaml /app/model-serve-spec.yaml
COPY metadata metadata

# create accessory directories
RUN mkdir -p /tmp 
RUN mkdir -p /models

ENTRYPOINT ["uv", "run", "uvicorn", "nnunet_serve.nnunet_serve:create_app"]