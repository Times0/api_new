# File: api/models.py
from engine.models import (
    Activity,
    Worker,
    Constraints,
    FilledActivity,
)
from pydantic import BaseModel
from typing import Optional


# API-specific models
class AutoSchedulingOptions(BaseModel):
    constraints: Optional[Constraints] = None


class ScheduleRequest(BaseModel):
    activities: list[Activity]
    workers: list[Worker]
    options: AutoSchedulingOptions


class Schedule(BaseModel):
    activities: list[FilledActivity]
    score: float
