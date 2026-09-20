"""Milvus Lite 冒烟 —— 验证 MilvusVectorStore 在本地文件模式下能不能跑通。

    python scripts/smoke_milvus_lite.py

★ 为什么需要这个脚本：
  AutoDL 的标准实例跑不了 Docker，Milvus standalone 起不来；而 SSH 反向隧道
  在 AutoDL 上又可能被 AllowTcpForwarding 挡住。Milvus Lite 是这两条路都走不通
  时的兜底 —— 向量库变成**一个本地文件**，零网络依赖。

★ 它验证的是**兼容性**，不是性能：
  milvus.py 用了一堆 standalone 才有的特性，Lite 不一定全支持。这个脚本把
  每一个都实际打一遍，而不是"应该没问题"。具体覆盖：

    · create_schema(auto_id=False, enable_dynamic_field=False)
    · VARCHAR + enable_analyzer + analyzer_params={"type": "chinese"}   ← 分词器
    · Function(function_type=FunctionType.BM25)                          ← 服务端 BM25
    · HNSW 稠密索引（Lite 可能只支持 FLAT/AUTOINDEX）
    · SPARSE_INVERTED_INDEX + metric_type="BM25" + DAAT_MAXSCORE
    · 标量字段 INVERTED 索引
    · upsert（不提供 sparse，由 BM25 Function 生成）
    · search(anns_field="sparse", data=[原文])                           ← BM25 查询
    · query(count(*)) / query_iterator / delete / run_analyzer

  任何一项不支持，都会在这里抛出明确的异常 —— 而不是等你传到 AutoDL 上才发现。

★ 输出刻意只用 ASCII 记号（[OK]/[FAIL]）：
  Windows 控制台默认 GBK 码页，打 ✓/✗ 会 UnicodeEncodeError 直接崩掉。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.infra.milvus import MilvusVectorStore  # noqa: E402
from rag.infra.vectorstore import VectorRecord  # noqa: E402

DB_PATH = ROOT / "data" / "vectors" / "_smoke_lite.db"
# ★ 分词器从环境变量读，默认 jieba：
#   Milvus Lite 只认 standard / jieba，**不认 chinese**（standalone 才认）。
#   想复现失败就 MILVUS_ANALYZER=chinese python scripts/smoke_milvus_lite.py
ANALYZER = os.environ.get("MILVUS_ANALYZER", "jieba")
COLLECTION = "smoke_lite"
DIM = 8          # 小维度，只为跑通链路，不测召回质量

_failures: list[str] = []


async def check(name: str, coro) -> bool:  # noqa: ANN001
    """跑一项检查，记录成败而不是立刻中断 —— 一次跑完能看到全部不兼容点。

    ★ 传进来的是协程对象（在调用处就 await 不了），这里统一 await。
      注意不能在里面再 asyncio.run()：main 本身就跑在事件循环里。
    """
    try:
        result = await coro
        print(f"[OK]   {name}" + (f" -> {result}" if result is not None else ""))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}")
        # pymilvus 的异常 str() 是一整个框，取第一行才有可读性
        first = str(exc).splitlines()[0].strip()[:300]
        print(f"       {type(exc).__name__}: {first}")
        _failures.append(name)
        return False


def make_records() -> list[VectorRecord]:
    texts = [
        "Milvus 是一个向量数据库，支持稠密向量和稀疏向量。",
        "BM25 是一种经典的全文检索算法，依赖词频和逆文档频率。",
        "RRF 融合可以把多路召回的结果合并成一个排序。",
    ]
    return [
        VectorRecord(
            chunk_id=1000 + i,
            dense=[0.1 * (i + 1)] * DIM,
            content=text,
            document_id=7,
            tenant_id=1,
            node_type="paragraph",
            lang="zh",
            content_hash=f"hash{i}",
            chunker_ver="v1",
            embed_model="smoke",
        )
        for i, text in enumerate(texts)
    ]


def _brief(hits) -> str:  # noqa: ANN001
    """检索结果只打印摘要 —— 这里验证的是连通性，不是召回质量。"""
    if not hits:
        raise AssertionError("返回空结果（能连上但没召回，可能是索引没生效）")
    return f"{len(hits)} 条, top1 chunk_id={hits[0].chunk_id} score={hits[0].score:.4f}"


def _clean(path: Path) -> None:
    """清掉上次的残留。

    ★ milvus-lite 3.2.1 把 uri 当成**目录**（data_dir），不是单文件 ——
      跑完这里躺着的是一个 `xxx.db/` 目录。所以不能只写 unlink()，
      否则第二次跑会直接 PermissionError: [WinError 5]。
      （老版本 milvus-lite 是单文件，两种形态都兼容一下。）
    """
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


async def main() -> int:
    _clean(DB_PATH)
    _clean(Path(str(DB_PATH) + ".lock"))
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(" Milvus Lite 冒烟（本地文件模式）")
    print(f" 数据文件 : {DB_PATH}")
    print("=" * 64)

    try:
        import pymilvus
    except ImportError as exc:
        print(f"\n[FAIL] 缺 pymilvus：{exc}\n")
        return 2
    try:
        import milvus_lite  # noqa: F401
    except ImportError:
        print("\n[FAIL] 没装 milvus-lite。先跑：uv pip install milvus-lite\n")
        return 2
    print(f"\n  pymilvus {pymilvus.__version__} / milvus-lite 已安装\n")

    # token 传空串 —— 本地文件没有鉴权。MilvusVectorStore 内部会转成 None。
    store = MilvusVectorStore(
        uri=str(DB_PATH), token="", collection=COLLECTION, dim=DIM, analyzer=ANALYZER
    )

    print("[1] 建集合（schema + BM25 Function + 索引）")
    ok = await check(
        "ensure_ready（含 HNSW / SPARSE_INVERTED / 标量 INVERTED）", store.ensure_ready()
    )
    if not ok:
        print("\n* 建集合失败 = Lite 不支持这套 schema，这条路不通。")
        return 1

    print("\n[2] 写入")
    await check("upsert 3 条（sparse 由服务端 BM25 生成）", store.upsert(make_records()))

    print("\n[3] 检索")
    await check("search_dense（COSINE，ef=128）",
                _wrap(store.search_dense([0.1] * DIM, top_k=3)))
    await check("search_sparse（BM25 传原文，服务端分词）",
                _wrap(store.search_sparse("全文检索算法", top_k=3)))

    print("\n[4] 运维接口")
    await check("count（count(*)）", store.count())
    await check("query_all_chunk_ids（query_iterator，对账用）",
                _count_ids(store))
    await check("run_analyzer（中文分词）", _analyze(store))

    print("\n[5] 进程内持久化（关掉重开，数据还在吗）")
    await store.aclose()
    store2 = MilvusVectorStore(
        uri=str(DB_PATH), token="", collection=COLLECTION, dim=DIM, analyzer=ANALYZER
    )
    ok2 = await check("重新打开后 ensure_ready", store2.ensure_ready())
    await check("重新打开后 count（数据还在）", store2.count())

    print("\n[6] 删除")
    # ★ 别在这里用上面已 close 的 client —— 会报 "should create connection first"，
    #   看起来像 Lite 不支持删除，其实是自己把连接关了还在用。
    if not ok2:
        await store2.ensure_ready()
    # ★ 这里断言删除条数，不只是打印。
    #   曾经 Lite 上恒返回 0（返回值的形状和 standalone 不一样），
    #   光看输出"-> 0"会以为删除没生效，其实数据已经没了。
    await check("delete_by_document（应删掉 3 条）", _expect(store2.delete_by_document(7, 1), 3))
    await check("删除后 count 归零", _expect(store2.count(), 0))
    await store2.aclose()

    print("\n" + "=" * 64)
    if _failures:
        print(f" X {len(_failures)} 项不兼容：")
        for f in _failures:
            print(f"     - {f}")
        print("\n   -> Lite 跑不了这套 schema，别在 AutoDL 上换 Lite。")
        print("      退回：修通 SSH 隧道，或给虚拟机加内存跑 standalone。")
    else:
        print(" OK 全部通过 —— Milvus Lite 可以替代 standalone，代码无需改动。")
    print("=" * 64)
    return 1 if _failures else 0


async def _wrap(coro):  # noqa: ANN001, ANN202
    return _brief(await coro)


async def _expect(coro, want: int) -> int:  # noqa: ANN001, ANN202
    """断言返回值 —— 只打印的话，"成功但报 0"这种假象会一直被放过去。"""
    got = await coro
    if got != want:
        raise AssertionError(f"期望 {want}，实际 {got}")
    return got


async def _count_ids(store) -> int:  # noqa: ANN001
    return len(await store.query_all_chunk_ids())


async def _analyze(store) -> str:  # noqa: ANN001
    return str(await store.analyze("向量数据库"))[:120]


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
