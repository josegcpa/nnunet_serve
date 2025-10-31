"""
Implementation of a nnUNet server API. 

Depends on ``model-serve-spec.yaml`` which should be specified in the directory
where nnunet_serve is utilized.
"""

import os
import threading
import time

import fastapi
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from nnunet_serve.logging_utils import get_logger
from nnunet_serve.nnunet_api_utils import CACHE
from nnunet_serve.nnunet_api import nnUNetAPI

logger = get_logger(__name__)

PORT = int(os.environ.get("UVICORN_PORT", "12345"))
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", "10"))


# global in-memory store for request timestamps per client IP
_rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()


async def expire_cache():
    CACHE.expire()


async def expire_cache_runner():
    while True:
        asyncio.create_task(expire_cache())
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    expire_cache_runner()
    yield


def create_app() -> fastapi.FastAPI:
    """
    Creates a FastAPI application.

    Returns:
        fastapi.FastAPI: FastAPI application.
    """
    app = fastapi.FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def _rate_limit_middleware(request: fastapi.Request, call_next):
        client_ip = request.client.host if request.client else "anonymous"
        now = time.time()
        with _rate_limit_lock:
            timestamps = _rate_limit_store.get(client_ip, [])
            timestamps = [t for t in timestamps if now - t < 60]
            if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
                raise fastapi.HTTPException(
                    status_code=429,
                    detail=f"Too many requests, limit is {MAX_REQUESTS_PER_MINUTE} per minute",
                )
            timestamps.append(now)
            _rate_limit_store[client_ip] = timestamps
        response = await call_next(request)
        return response

    nnunet_api = nnUNetAPI(app)
    nnunet_api.init_api()

    return nnunet_api.app


if __name__ == "__main__":
    uvicorn.run(
        "nnunet_serve.nnunet_serve:create_app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
