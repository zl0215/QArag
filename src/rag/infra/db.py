"""PostgreSQL 连接管理。

★ 三个必须做对的地方：

1. **`pool_pre_ping=True`** —— 虚拟机挂起/网络抖动后，池里的死连接会被自动剔除。
   不加这个，第一次请求必报 `ConnectionDoesNotExistError`。

2. **`expire_on_commit=False`** —— 默认 True 时 commit 后访问任何属性都会触发
   一次懒加载 SELECT，在 async 下直接抛 `MissingGreenlet`。这是 async SQLAlchemy
   最常见的坑。

3. **禁用 SQLite 跑集成测试** —— `FOR UPDATE SKIP LOCKED`、JSONB、部分唯一索引
   都不支持。测试要么用真 Postgres，要么用内存实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from rag.core.logging import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, url: str, *, echo: bool = False, pool_size: int = 10) -> None:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        self.url = url
        self._engine = create_async_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=pool_size,
            pool_recycle=1800,
            connect_args={"server_settings": {"application_name": "rag-agent"}},
        )
        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, autoflush=False
        )
        self._checkpointer = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator:
        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_all(self) -> None:
        """建表。生产应换成 Alembic 迁移，一期用 create_all 足够。"""
        from rag.infra.models import Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("db.schema_ready")

    async def setup_checkpointer(self) -> None:
        """初始化 LangGraph 的 Postgres checkpointer。

        ★ 用官方 `langgraph-checkpoint-postgres` 而不是社区 MySQL 版：
          前者由 LangChain 官方维护并跟随 LangGraph 版本发布，后者是单人维护，
          且与 MySQL 9.6+ 不兼容（生成列里不能用 MD5 做默认值）。
        """
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError:  # pragma: no cover
            logger.warning("db.checkpointer_unavailable", hint="未安装 langgraph-checkpoint-postgres")
            return

        # checkpointer 需要 psycopg 的原生连接串，不是 SQLAlchemy 的 +asyncpg 形式
        dsn = (
            self.url.replace("postgresql+asyncpg://", "postgresql://")
            .replace("postgresql+psycopg://", "postgresql://")
        )
        saver = AsyncPostgresSaver.from_conn_string(dsn)
        self._checkpointer = await saver.__aenter__()
        await self._checkpointer.setup()
        logger.info("db.checkpointer_ready")

    @property
    def checkpointer(self):  # noqa: ANN201
        return self._checkpointer

    async def healthcheck(self) -> bool:
        from sqlalchemy import text

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.warning("db.healthcheck_failed", exc_info=True)
            return False

    async def dispose(self) -> None:
        if self._checkpointer is not None:
            try:
                await self._checkpointer.__aexit__(None, None, None)
            except Exception:  # pragma: no cover
                logger.debug("db.checkpointer_close_failed", exc_info=True)
            self._checkpointer = None
        await self._engine.dispose()
        logger.info("db.disposed")
