"""向 PostgreSQL 写入一组可重复执行的 V2 教务 Demo 数据。"""

from __future__ import annotations

import asyncio
import datetime as dt

from sqlalchemy.dialects.postgresql import insert

from rag.core.config import get_settings
from rag.infra.db import Database
from rag.infra.models import AcademicCourse, StudentExam, StudentGrade, StudentSchedule

STUDENT_ID = "demo-student"


async def main() -> int:
    settings = get_settings()
    if settings.repository_backend != "postgres":
        raise SystemExit("seed_academic_demo 只写 PostgreSQL；请设置 REPOSITORY_BACKEND=postgres")

    db = Database(settings.database_url)
    await db.create_all()
    async with db.session() as session:
        await _upsert(
            session,
            AcademicCourse,
            [
                {
                    "tenant_id": 1,
                    "course_id": "AI-101-A",
                    "course_code": "AI101",
                    "name": "人工智能导论",
                    "credits": 3,
                    "category": "人工智能",
                    "department": "计算机学院",
                    "term": "2027-spring",
                    "instructor": "张老师",
                    "available_seats": 12,
                    "meeting_times": [{
                        "weekday": 1, "start_time": "09:00", "end_time": "10:40",
                        "weeks": list(range(1, 17)), "location": "A201",
                    }],
                },
                {
                    "tenant_id": 1,
                    "course_id": "ML-201-B",
                    "course_code": "ML201",
                    "name": "机器学习",
                    "credits": 3,
                    "category": "人工智能",
                    "department": "计算机学院",
                    "term": "2027-spring",
                    "instructor": "李老师",
                    "available_seats": 8,
                    "meeting_times": [{
                        "weekday": 2, "start_time": "14:00", "end_time": "15:40",
                        "weeks": list(range(1, 17)), "location": "B301",
                    }],
                },
            ],
            ["tenant_id", "course_id"],
        )
        await _upsert(
            session,
            StudentGrade,
            [
                {
                    "tenant_id": 1, "student_id": STUDENT_ID,
                    "course_code": "MATH101", "course_name": "高等数学",
                    "credits": 60, "score": 85, "grade_point": 3.7,
                    "status": "passed", "term": "2025-fall",
                },
                {
                    "tenant_id": 1, "student_id": STUDENT_ID,
                    "course_code": "CS101", "course_name": "程序设计",
                    "credits": 60, "score": 90, "grade_point": 4.0,
                    "status": "passed", "term": "2026-spring",
                },
            ],
            ["tenant_id", "student_id", "course_code", "term"],
        )
        await _upsert(
            session,
            StudentSchedule,
            [{
                "tenant_id": 1, "student_id": STUDENT_ID,
                "course_id": "MATH-ADV-A", "course_code": "MATH301",
                "course_name": "高等数学进阶", "term": "2027-spring",
                "meeting_times": [{
                    "weekday": 1, "start_time": "08:50", "end_time": "10:25",
                    "weeks": list(range(1, 17)), "location": "A101",
                }],
            }],
            ["tenant_id", "student_id", "course_id", "term"],
        )
        await _upsert(
            session,
            StudentExam,
            [{
                "tenant_id": 1, "student_id": STUDENT_ID,
                "course_code": "CS101", "course_name": "程序设计",
                "term": "2026-spring", "exam_type": "期末考试",
                "start_at": dt.datetime(2027, 1, 8, 9, 0, tzinfo=dt.UTC),
                "end_at": dt.datetime(2027, 1, 8, 11, 0, tzinfo=dt.UTC),
                "location": "教学楼 C302", "seat": "18", "status": "scheduled",
            }],
            ["tenant_id", "student_id", "course_code", "term", "exam_type"],
        )
        await session.commit()
    await db.dispose()
    print(f"V2 Demo 数据已写入，student_id={STUDENT_ID}")
    return 0


async def _upsert(session, model, rows: list[dict], keys: list[str]) -> None:  # noqa: ANN001
    for row in rows:
        statement = insert(model).values(**row)
        updates = {key: value for key, value in row.items() if key not in keys}
        await session.execute(statement.on_conflict_do_update(index_elements=keys, set_=updates))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
