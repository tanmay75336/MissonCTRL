from typing import Any, Literal

from pydantic import BaseModel, Field


ResearchEventType = Literal[
    "MISSION_STARTED",
    "RESEARCH_PLAN_CREATED",
    "OBJECTIVE_SELECTED",
    "QUESTION_CREATED",
    "STRATEGY_SELECTED",
    "RESEARCH_QUESTION",
    "SEARCH_STARTED",
    "SEARCH_COMPLETED",
    "EVIDENCE_EVALUATED",
    "COVERAGE_UPDATED",
    "GAP_IDENTIFIED",
    "NEXT_ACTION_SELECTED",
    "EVIDENCE_COLLECTED",
    "EVIDENCE_REJECTED",
    "DECISION",
    "STRATEGY_SWITCHED",
    "CLAIM_VALIDATED",
    "CONTRADICTION_FOUND",
    "SYNTHESIS_STARTED",
    "MISSION_COMPLETED",
    "MISSION_STOPPED",
]


class ResearchEvent(BaseModel):
    """A concise operational event safe to expose to API clients."""

    type: ResearchEventType
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
