"""Async Redis access for the API (the worker uses the sync registry)."""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis

from ..config import Settings, get_settings

QUEUE = "research:queue"


def run_key(run_id: str) -> str:
    return f"research:run:{run_id}"


def trace_key(run_id: str) -> str:
    return f"research:trace:{run_id}"


@lru_cache
def _client_for(url: str) -> aioredis.Redis:
    return aioredis.Redis.from_url(url, decode_responses=True)


def get_redis_client(settings: Settings | None = None) -> aioredis.Redis:
    settings = settings or get_settings()
    return _client_for(settings.redis_url)


async def redis_ping(client: aioredis.Redis) -> bool:
    try:
        return bool(await client.ping())
    except Exception:
        return False
