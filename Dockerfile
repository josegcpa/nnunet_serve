FROM python:3.11.4-slim-bullseye
USER root
WORKDIR /app

# install environment
COPY ../requirements.txt requirements.txt
COPY ../docker-installations.sh docker-installations.sh
RUN bash docker-installations.sh

# create accessory directories
RUN mkdir /model && \
    mkdir -p /data && \
    mkdir -p /data/input && \
    mkdir -p /data/output && \
    mkdir -p utils

# copy scripts
COPY ../utils/utils.py utils/
COPY ../utils/pydicom_seg_writers.py utils/
COPY ../utils/entrypoint-prod.py utils/
COPY ../utils/entrypoint.py utils/
RUN chown root -R utils
ENTRYPOINT ["python", "utils/entrypoint-prod.py","-o /data/output","-m /model"]