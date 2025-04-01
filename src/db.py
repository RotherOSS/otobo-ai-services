import asyncpg
import asyncio
from typing import Optional
from loguru import logger

_pg_conn: Optional[asyncpg.Connection] = None


async def init_pg_connection(dsn: str, retries=5, delay=5):
    global _pg_conn
    if _pg_conn is None:
        await asyncio.sleep(delay)
        for attempt in range(1, retries + 1):
            try:
                _pg_conn = await asyncpg.connect(dsn)
                logger.success("Connected to Postgres.")
                return
            except Exception as e:
                logger.warning(f"Postgres not ready (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    raise
                await asyncio.sleep(delay)


async def close_pg_connection():
    global _pg_conn
    if _pg_conn is not None:
        await _pg_conn.close()
        _pg_conn = None


def get_pg_connection() -> asyncpg.Connection:
    if _pg_conn is None:
        raise RuntimeError("Postgres connection not initialized.")
    return _pg_conn
