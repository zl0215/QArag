"""Pi 风格的最小 Agent：只负责构造消息并让模型选择下一步。"""

from __future__ import annotations

import json
import re
from typing import Any

from rag.agent.prompts import PI_AGENT_SYSTEM_PROMPT
from rag.providers.base import LLMChatResponse, LLMClient


class PiAgent:
    """薄模型适配层；Loop、工具执行和错误处理都由 Harness 负责。"""

    def __init__(self, *, llm: LLMClient, tool_definitions: list[dict[str, Any]]) -> None:
        self.llm = llm
        self.tool_definitions = tool_definitions

    def initial_messages(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        clean_history = [
            {"role": item["role"], "content": item["content"]}
            for item in (history or [])[-12:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        return [
            {"role": "system", "content": PI_AGENT_SYSTEM_PROMPT},
            *clean_history,
            {"role": "user", "content": question},
        ]

    async def step(self, messages: list[dict[str, Any]]) -> LLMChatResponse:
        return await self.llm.achat(
            messages,
            tools=self.tool_definitions,
            temperature=0.0,
        )

    @staticmethod
    def requires_knowledge(question: str) -> bool:
        """Prompt 的硬约束兜底，避免学校规定被一次偶发的直答绕过。"""
        text = question.casefold()
        explicit_corpus = re.search(
            r"知识库|已上传|上传的.{0,8}(?:文档|资料|论文)|根据.{0,12}(?:文档|资料|论文)|"
            r"(?:文档|论文|手册|规定)中",
            text,
        )
        school_scope = re.search(r"学校|本校|校规|教务|学籍|培养方案", text)
        policy_topic = re.search(
            r"规定|要求|重修|补考|毕业|学分|选课|退课|考试|成绩|奖学金|处分|请假|宿舍",
            text,
        )
        graduation_credit = "毕业" in text and "学分" in text
        return bool(explicit_corpus or (school_scope and policy_topic) or graduation_credit)

    @staticmethod
    def assistant_message(response: LLMChatResponse) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or None,
        }
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in response.tool_calls
            ]
        return message


__all__ = ["PiAgent"]
