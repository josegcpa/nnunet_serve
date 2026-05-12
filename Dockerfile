FROM python:3.11.4-slim-bullseye
LABEL org.opencontainers.image.authors="Jose Almeida <jose.almeida@research.fchampalimaud.org>"

USER root
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=12345
ENV NNUNET_OUTPUT_DIR=/data/nnunet
ENV MODEL_DIR=/models
ENV MODEL_SERVE_SPEC=/app/model-serve-spec.yaml

# install internal dependencies
RUN pip install uv pip && \
    uv clean && \
    apt update && \
    apt install libgl1 libglib2.0-0 git curl -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

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

COPY metadata /app/metadata
COPY model-serve-spec.yaml /tmp/model-serve-spec.yaml
RUN sed "s#model_folder: .*#model_folder: $MODEL_DIR#" /tmp/model-serve-spec.yaml > $MODEL_SERVE_SPEC
RUN echo $MODEL_SERVE_SPEC

# create accessory directories
RUN mkdir -p /tmp 
RUN mkdir -p $MODEL_DIR
RUN mkdir -p $NNUNET_OUTPUT_DIR

# Downloads total segmentator models if necessary 
RUN uv run nnunet-validate-metadata

EXPOSE 12345

ENTRYPOINT ["uv", "run", "--project", "/app"]
CMD ["python", "-m", "uvicorn", "nnunet_serve.nnunet_serve_api:create_app", "--host", "0.0.0.0", "--port", "12345"]