from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.agent.evaluator import EvaluationResult, NextActionDecision
from app.agent.events import ResearchEvent
from app.agent.research_plan import Claim, CoverageMap, ResearchPlan, ResearchSearchRecord


class MissionRequest(BaseModel):
    mission: str = Field(min_length=1)
    plan: str = ""
    search_budget: int = Field(default=4, ge=1, le=8)

    @field_validator("mission")
    @classmethod
    def mission_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Mission cannot be empty.")
        return value


class MissionResponse(BaseModel):
    mission: str
    answer: str
    facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    sources: list[Any] = Field(default_factory=list)
    strategy_history: list[str] = Field(default_factory=list)
    searches_used: int
    remaining_searches: int
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    decisions: list[NextActionDecision] = Field(default_factory=list)
    events: list[ResearchEvent] = Field(default_factory=list)
    research_plan: ResearchPlan | None = None
    coverage: CoverageMap | None = None
    claims: list[Claim] = Field(default_factory=list)
    search_records: list[ResearchSearchRecord] = Field(default_factory=list)
