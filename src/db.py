import aiomysql
import asyncio
from typing import Optional, Sequence, Any
from loguru import logger

# Global variable holding the async connection pool.
_db_pool: Optional["_PoolWrapper"] = None


class _ConnWrapper:
    """Adapts an aiomysql connection to the asyncpg-style call shape (execute/fetch/
    fetchrow/executemany) that the rest of the codebase is written against."""

    def __init__(self, conn):
        self._conn = conn

    async def execute(self, query: str, *args):
        async with self._conn.cursor() as cur:
            await cur.execute(query, args)

    async def fetch(self, query: str, *args):
        async with self._conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, args)
            return await cur.fetchall()

    async def fetchrow(self, query: str, *args):
        async with self._conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(query, args)
            return await cur.fetchone()

    async def executemany(self, query: str, args_list: Sequence[Sequence[Any]]):
        async with self._conn.cursor() as cur:
            await cur.executemany(query, args_list)


class _AcquireCtx:
    def __init__(self, pool: aiomysql.Pool):
        self._pool = pool
        self._conn_ctx = None

    async def __aenter__(self) -> _ConnWrapper:
        self._conn_ctx = self._pool.acquire()
        conn = await self._conn_ctx.__aenter__()
        return _ConnWrapper(conn)

    async def __aexit__(self, exc_type, exc, tb):
        return await self._conn_ctx.__aexit__(exc_type, exc, tb)


class _PoolWrapper:
    def __init__(self, pool: aiomysql.Pool):
        self._pool = pool

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._pool)

    async def close(self):
        self._pool.close()
        await self._pool.wait_closed()


async def init_db_pool(host: str, port: int, user: str, password: str, database: str, retries=5, delay=5):
    """
    Initializes a global async MariaDB connection pool.

    Retries connection `retries` times with `delay` seconds between attempts.
    Useful for waiting on DB readiness in containerized environments.
    """
    global _db_pool
    if _db_pool is None:
        await asyncio.sleep(delay)  # Initial delay (e.g., give DB time to start)
        for attempt in range(1, retries + 1):
            try:
                pool = await aiomysql.create_pool(
                    host=host,
                    port=int(port),
                    user=user,
                    password=password,
                    db=database,
                    autocommit=True,
                )
                _db_pool = _PoolWrapper(pool)
                logger.success("Connection pool to MariaDB initialized.")
                return
            except Exception as e:
                logger.warning(f"MariaDB pool not ready (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    raise  # Give up after last retry
                await asyncio.sleep(delay)


async def close_db_pool():
    """
    Closes the global MariaDB connection pool if initialized.
    """
    global _db_pool
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None


def get_db_pool() -> _PoolWrapper:
    """
    Returns the global connection pool.
    Raises if accessed before initialization.
    """
    if _db_pool is None:
        raise RuntimeError("MariaDB pool not initialized.")
    return _db_pool
