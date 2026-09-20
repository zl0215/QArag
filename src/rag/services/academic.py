"""教务只读查询服务。

这一层只返回事实或确定性的时间冲突计算，不替 Agent 做选课、毕业判断等决策。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Any

from rag.infra.repository import Repository


class AcademicService:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    async def search_courses(
        self,
        *,
        tenant_id: int,
        keyword: str | None = None,
        term: str | None = None,
        department: str | None = None,
        category: str | None = None,
        min_credits: float | None = None,
        max_credits: float | None = None,
        limit: int = 20,
        course_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.repository.list_courses(
            tenant_id=tenant_id,
            keyword=keyword,
            term=term,
            department=department,
            category=category,
            min_credits=min_credits,
            max_credits=max_credits,
            limit=limit,
            course_ids=course_ids,
        )
        return [_public_record(row) for row in rows]

    async def query_grades(
        self,
        student_id: str,
        *,
        tenant_id: int,
        term: str | None = None,
        status: str | None = None,
        course_code: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.repository.list_grades(
            student_id,
            tenant_id=tenant_id,
            term=term,
            status=status,
            course_code=course_code,
        )
        return [_public_record(row) for row in rows]

    async def query_schedule(
        self,
        student_id: str,
        *,
        tenant_id: int,
        term: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.repository.list_schedule(
            student_id,
            tenant_id=tenant_id,
            term=term,
        )
        return [_public_record(row) for row in rows]

    async def query_exams(
        self,
        student_id: str,
        *,
        tenant_id: int,
        term: str | None = None,
        course_code: str | None = None,
        from_at: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.repository.list_exams(
            student_id,
            tenant_id=tenant_id,
            term=term,
            course_code=course_code,
            from_at=from_at,
        )
        result = []
        for row in rows:
            item = _public_record(row)
            item["start_at"] = row.start_at.isoformat()
            item["end_at"] = row.end_at.isoformat()
            result.append(item)
        return result

    async def check_schedule_conflict(
        self,
        student_id: str,
        course_ids: list[str],
        *,
        tenant_id: int,
        term: str | None = None,
    ) -> list[dict[str, Any]]:
        candidates = await self.search_courses(
            tenant_id=tenant_id,
            term=term,
            course_ids=course_ids,
            limit=max(len(course_ids), 1),
        )
        current = await self.query_schedule(
            student_id,
            tenant_id=tenant_id,
            term=term,
        )
        by_id = {course["course_id"]: course for course in candidates}
        output: list[dict[str, Any]] = []

        for course_id in course_ids:
            course = by_id.get(course_id)
            if course is None:
                output.append({
                    "course_id": course_id,
                    "found": False,
                    "has_conflict": None,
                    "conflicts": [],
                })
                continue
            conflicts = []
            for existing in current:
                overlaps = _meeting_overlaps(
                    course.get("meeting_times") or [],
                    existing.get("meeting_times") or [],
                )
                if overlaps:
                    conflicts.append({
                        "with_course_id": existing["course_id"],
                        "with_course_code": existing["course_code"],
                        "with_course_name": existing["course_name"],
                        "overlaps": overlaps,
                    })
            output.append({
                "course_id": course_id,
                "course_code": course["course_code"],
                "course_name": course["name"],
                "found": True,
                "has_conflict": bool(conflicts),
                "conflicts": conflicts,
            })
        return output


def _meeting_overlaps(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for candidate in left:
        for existing in right:
            if int(candidate.get("weekday") or 0) != int(existing.get("weekday") or 0):
                continue
            left_weeks = set(candidate.get("weeks") or [])
            right_weeks = set(existing.get("weeks") or [])
            if left_weeks and right_weeks and left_weeks.isdisjoint(right_weeks):
                continue
            try:
                start = max(
                    dt.time.fromisoformat(str(candidate["start_time"])),
                    dt.time.fromisoformat(str(existing["start_time"])),
                )
                end = min(
                    dt.time.fromisoformat(str(candidate["end_time"])),
                    dt.time.fromisoformat(str(existing["end_time"])),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("课程时间数据格式错误，必须使用 HH:MM") from exc
            if start < end:
                overlaps.append({
                    "weekday": int(candidate["weekday"]),
                    "start_time": start.isoformat(timespec="minutes"),
                    "end_time": end.isoformat(timespec="minutes"),
                    "weeks": sorted(left_weeks & right_weeks) if left_weeks and right_weeks else [],
                })
    return overlaps


def _public_record(record: Any) -> dict[str, Any]:
    """Tool Result 不向模型重复暴露租户 ID 和学生 ID。"""
    item = asdict(record)
    item.pop("tenant_id", None)
    item.pop("student_id", None)
    return item


__all__ = ["AcademicService"]
