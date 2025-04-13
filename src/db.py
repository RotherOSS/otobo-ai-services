import asyncpg
import asyncio
from typing import Optional
from loguru import logger

# Global variable holding the async connection pool.
_pg_pool: Optional[asyncpg.Pool] = None


async def init_pg_pool(dsn: str, retries=5, delay=5):
    """
    Initializes a global async Postgres connection pool.

    Retries connection `retries` times with `delay` seconds between attempts.
    Useful for waiting on DB readiness in containerized environments.
    """
    global _pg_pool
    if _pg_pool is None:
        await asyncio.sleep(delay)  # Initial delay (e.g., give DB time to start)
        for attempt in range(1, retries + 1):
            try:
                _pg_pool = await asyncpg.create_pool(dsn)
                logger.success("Connection pool to Postgres initialized.")
                return
            except Exception as e:
                logger.warning(f"Postgres pool not ready (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    raise  # Give up after last retry
                await asyncio.sleep(delay)


async def close_pg_pool():
    """
    Closes the global Postgres connection pool if initialized.
    """
    global _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None


def get_pg_pool() -> asyncpg.Pool:
    """
    Returns the global connection pool.
    Raises if accessed before initialization.
    """
    if _pg_pool is None:
        raise RuntimeError("Postgres pool not initialized.")
    return _pg_pool
