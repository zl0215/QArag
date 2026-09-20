"""全局配置。

所有可调参数集中在这里，通过环境变量或 .env 覆盖。
字段名小写，pydantic-settings 自动映射到同名大写环境变量（POSTGRES_HOST 等）。
"""

from __future__ import annotations

import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # pydantic v2.9+ 才导出 ValidationInfo 的稳定路径
    from pydantic import ValidationInfo
except ImportError:  # pragma: no cover
    from pydantic_core.core_schema import ValidationInfo  # type: ignore[assignment]


def _allows_none(annotation: Any) -> bool:
    """判断类型注解是否允许 None。

    ★ 必须同时判 `typing.Union` 和 `types.UnionType`：
      `Optional[int]` 的 origin 是前者，`int | None` 的 origin 是后者，
      两者是**不同的对象**（PEP 604 的 `|` 在 3.10+ 才产生 UnionType）。
      只判其中一个，另一个就会静默返回 False —— 表现为"校验器没生效"，
      但没有任何报错，非常难查。
    """
    if annotation is type(None):
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(arg is type(None) for arg in get_args(annotation))
    return False

EmbedProvider = Literal["local", "api", "hash"]
RerankProvider = Literal["local", "api", "none"]
LLMProvider = Literal["openai", "none"]
# milvus = standalone（走网络，需要 Docker）或 Milvus Lite（本地文件），
#          靠 MILVUS_URI 的格式区分，不是两个后端
# memory = 纯内存，进程退出即丢，只用于单测和一次性冒烟
VectorBackend = Literal["milvus", "memory"]
# openai = OpenAI 兼容 /embeddings（vLLM、Xinference、硅基流动…）
# tei    = HuggingFace text-embeddings-inference 原生 /embed
EmbedApiStyle = Literal["openai", "tei"]
# cohere = {"results":[{"index","relevance_score"}]}（Cohere/Jina/硅基流动）
# tei    = [{"index","score"}]（text-embeddings-inference）
RerankApiStyle = Literal["cohere", "tei"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- 应用 ----------------
    app_env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    api_port: int = 8000

    # ---------------- PostgreSQL ----------------
    # postgres = 正文 + 任务表 + LangGraph checkpoint（生产 / AutoDL 自装 PG）
    # memory   = 进程内，重启即丢。**只用于"先跑起来看看"**，
    #            因为它同时意味着没有任务队列、没有 checkpoint 续跑。
    repository_backend: Literal["postgres", "memory"] = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag"
    postgres_password: SecretStr = SecretStr("rag_dev_pw")
    postgres_db: str = "rag"
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")

    # ---------------- Milvus ----------------
    # ★ 同一个 VECTOR_BACKEND=milvus 有两种形态，靠这个字段的**格式**区分：
    #     http://host:19530  → standalone / 集群（走网络，需要 Docker）
    #     ./data/milvus.db   → Milvus Lite（本地目录，零网络依赖，uv sync --extra lite）
    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "rag_chunks"
    # standalone 用 chinese；★ Milvus Lite 只认 jieba / standard
    milvus_analyzer: str = "chinese"

    # ---------------- 向量后端 ----------------
    # memory = 纯内存实现，Windows 本地开发 / 单测用，无需 Docker
    vector_backend: VectorBackend = "memory"

    @property
    def milvus_is_lite(self) -> bool:
        """uri 不是 http(s) 就当成 Milvus Lite 的本地目录。

        ★ 需要这个判断是因为 Lite 有两条 standalone 没有的硬约束：
          ① 一个 data_dir 同时只能被**一个进程**打开（文件锁）——
             所以 API 和 worker 不能同时用它
          ② 分词器只认 jieba / standard
        这两条都得在启动时给出明确提示，而不是等一个诡异的锁错误。
        """
        return not self.milvus_uri.startswith(("http://", "https://"))

    # ---------------- Embedding ----------------
    embed_provider: EmbedProvider = "local"
    embed_model_path: str = ""
    embed_model_id: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    # 留空则从模型 config 的 max_position_embeddings 推断
    embed_max_tokens: int | None = None
    # bge-m3 的多语言检索不要求 instruction；额外加中文前缀反而会污染英文译文查询。
    embed_query_instruction: str = ""
    embed_batch_size: int = 16
    # cpu / cuda / cuda:0 —— 有 GPU 时设 cuda，吞吐差一个数量级
    embed_device: str = "cpu"
    embed_api_base: str = ""
    embed_api_key: SecretStr = SecretStr("")
    embed_api_model: str = "BAAI/bge-m3"
    embed_api_style: EmbedApiStyle = "openai"
    embed_timeout_seconds: int = 60
    # 启动时是否阻塞等待模型加载完。本地模型冷启动几十秒，
    # 设 false 则 /healthz 先返回、模型在后台加载（API 更快就绪但首个请求会等）
    embed_warmup_on_startup: bool = True

    # ---------------- Reranker ----------------
    rerank_provider: RerankProvider = "none"
    rerank_model_path: str = ""
    rerank_api_base: str = ""
    rerank_api_key: SecretStr = SecretStr("")
    rerank_api_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_style: RerankApiStyle = "cohere"
    rerank_timeout_seconds: int = 60

    # ---------------- LLM ----------------
    llm_provider: LLMProvider = "none"
    llm_api_base: str = ""
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "deepseek-chat"
    llm_model_cheap: str = "deepseek-chat"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 120

    # ---------------- 分块 ----------------
    chunk_target_tokens: int = 256
    chunk_max_tokens: int = 384
    chunk_overlap_tokens: int = 48
    parent_target_tokens: int = 1024
    # 分块逻辑或参数变更时必须 bump，否则增量索引会错误复用旧块
    chunker_version: str = "v2-layout-multilingual"

    # ---------------- 检索 ----------------
    retrieve_dense_top_k: int = 50
    retrieve_sparse_top_k: int = 50
    retrieve_fused_top_k: int = 60
    retrieve_final_top_k: int = 5
    rrf_k: int = 60
    retrieve_score_threshold: float = 0.0
    context_max_tokens: int = 4000
    # 跨语言查询扩展：查询语言与语料语言不一致时，追加一路译文召回。
    # 实测（中文提问 / 英文语料）chunk recall@5 0.375 → 0.500，
    # 词法通道召回由 0.175 恢复到 0.650。详见 services/translate.py。
    retrieve_translate: bool = True
    retrieve_translate_target: str = "en"
    # 留空则用 llm_model_cheap —— 改写任务不需要强模型
    retrieve_translate_model: str = ""

    # ---------------- Agent ----------------
    agent_max_retries: int = 2
    agent_recursion_limit: int = 50
    agent_summary_after_turns: int = 20
    # V2 多工具任务 Loop。与旧 LangGraph 的 rewrite 次数分开配置。
    agent_max_iterations: int = 8
    agent_tool_timeout_seconds: int = 120

    # ---------------- Neo4j ----------------
    enable_graph: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("rag_dev_pw")
    graph_max_triplets_per_chunk: int = 30
    graph_expand_limit: int = 20

    # ---------------- 上传 ----------------
    upload_max_mb: int = 50
    upload_dir: Path = Path("./data/uploads")

    # ---------------- Worker ----------------
    worker_poll_interval_seconds: float = 2.0
    worker_batch_size: int = 4

    @field_validator("*", mode="before")
    @classmethod
    def _empty_string_means_unset(cls, value: Any, info: ValidationInfo) -> Any:
        """把空字符串当作"未设置"。

        ★ 这条规则来自一个真实的坑：`.env` 里写 `EMBED_MAX_TOKENS=`（留空表示用默认值）
          或 docker-compose 传 `FOO: ${FOO:-}` 时，值会是空串而不是缺失。
          pydantic 会把空串塞给 `int | None` 解析，直接抛 ValidationError，
          而报错信息完全看不出是"某个环境变量留空了"。

        只对**允许 None** 的字段生效 —— 非可选字段留空仍应报错，
        否则 `POSTGRES_HOST=` 会静默变成 None 而不是提示配置缺失。
        """
        if value == "" and _allows_none(cls.model_fields[info.field_name].annotation):
            return None
        return value

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """异步 SQLAlchemy 连接串。密码做 URL 编码，防止特殊字符破坏 DSN。"""
        if self.database_url_override:
            return self.database_url_override
        pwd = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_dev(self) -> bool:
        return self.app_env in ("dev", "test")

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
