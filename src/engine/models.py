# File: engine/models.py
import dataclasses
import enum
from typing import Generic, Optional, TypeVar, TypedDict

from pydantic import BaseModel


# Core Engine Models
class SkillLevelRequirement(BaseModel):
    skillLevel: int
    minWorkers: int


class WorkstationLevel(str, enum.Enum):
    LEVEL_0 = "LEVEL0"
    LEVEL_1 = "LEVEL1"
    LEVEL_2 = "LEVEL2"
    LEVEL_3 = "LEVEL3"
    LEVEL_4 = "LEVEL4"


class StrainLevel(str, enum.Enum):
    LOW = "LOW"
    MILD = "MILD"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class ConstraintData(BaseModel):
    expectedNbWorker: Optional[int]
    skillLevelRequirements: Optional[list[SkillLevelRequirement]]
    strainLevel: Optional[StrainLevel]

    def __init__(self, **data):
        super().__init__(**data)
        if self.skillLevelRequirements is None:
            self.skillLevelRequirements = []


class Activity(BaseModel):
    id: str
    workstationId: str
    constraintData: ConstraintData
    startDateTime: str
    endDateTime: str


class Worker(BaseModel):
    id: str
    name: str


T = TypeVar("T")


class Constraint(BaseModel, Generic[T]):
    weight: float
    constraintData: Optional[T] = None


class LegalHoursConstraint(BaseModel):
    workingHoursDay: int
    workingHoursWeek: int
    restHoursWeek: int
    weeklyActivities: Optional[list[Activity]] = None  # Activities from the entire week for reference


class ActivityShift(TypedDict):
    activityId: str
    shiftId: str


class MaxShiftsPerDayConstraint(BaseModel):
    workerActivitiesMatrix: dict[str, list[list[ActivityShift]]]  # workerId -> list of shifts per day
    shiftTransitionMatrix: dict[str, dict[str, bool]]  # shiftId -> shiftId -> can transition
    shiftIds: list[str]  # list of shift IDs
    scheduleDays: list[str]  # list of dates


class TrainingInterval(BaseModel):
    startDate: str  # ISO format date string
    endDate: str  # ISO format date string


class Constraints(BaseModel):
    spreadWorkers: Constraint[None]
    localization: Constraint[dict[str, dict[str, bool]]]
    preAssignedWorkers: Constraint[dict[str, list[str]]]
    availability: Constraint[dict[str, list[str]]]
    workerLocalization: Constraint[None]
    skillLevels: Constraint[dict[str, dict[str, WorkstationLevel]]]
    maxShiftsPerDay: Constraint[MaxShiftsPerDayConstraint]
    trainings: Constraint[dict[str, list[TrainingInterval]]]


# Type aliases for engine internal use
preAssignedWorkersPerActivityT = dict[str, list[str]]  # activityId → workerIds
workerWorkstationsLevelMatrixT = dict[str, dict[str, WorkstationLevel]]  # workstationId → level → worker IDs
localizationMatrix = dict[str, dict[str, bool]]  # workerId → workstationId → bool isAssigned
workerAvailibilityMatrix = dict[str, list[str]]  # workerId -> vacationDays
workerTrainingsMatrix = dict[str, list[TrainingInterval]]  # workerId -> list of training intervals


# Engine-specific models for results


class FilledActivity(BaseModel):
    id: str
    workstationId: str
    constraintData: ConstraintData
    startDateTime: str
    endDateTime: str
    assignedWorkerIds: list[str]


@dataclasses.dataclass
class ScheduleResult:
    success: bool
    error_message: Optional[str]
    assignments: dict[tuple[str, str], bool]
    statistics: dict[str, any]  # or tools stats like walltime branches etc
    penalty: float
    score: float


@dataclasses.dataclass
class SchedulingResultWithFilledActivities(ScheduleResult):
    filled_activities: list[FilledActivity]
