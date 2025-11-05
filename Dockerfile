FROM python:3.11.4-slim-bullseye
USER root
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=50422
ENV NNUNET_OUTPUT_DIR=/data/nnunet

# install internal dependencies
RUN pip install uv pip && \
    uv pip install --system setuptools wheel && \
    uv pip install --system -r requirements.txt && \
    uv clean && \
    apt update && \
    apt install libgl1 libglib2.0-0 git -y && \
    apt clean

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
RUN mkdir -p $NNUNET_OUTPUT_DIR

# downloads total segmentator models if necessary 
RUN uv run nnunet-validate-metadata

ENTRYPOINT ["uv", "run" ,"--project", "/app"]