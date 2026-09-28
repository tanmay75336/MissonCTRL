from pydantic import BaseModel, Field


class MissionPlan(BaseModel):
    objective: str

    explicit_constraints: list[str] = Field(
        default_factory=list
    )

    assumptions: list[str] = Field(
        default_factory=list
    )

    unknowns: list[str] = Field(
        default_factory=list
    )

    research_questions: list[str] = Field(
        default_factory=list
    )

    required_tools: list[str] = Field(
        default_factory=list
    )

    success_criteria: list[str] = Field(
        default_factory=list
    )


class MissionState(BaseModel):
    mission: str

    plan: MissionPlan | None = None

    completed_questions: list[str] = Field(
        default_factory=list
    )

    findings: list[str] = Field(
        default_factory=list
    )

    evidence: list[dict] = Field(
        default_factory=list
    )

    sources: list[str] = Field(
        default_factory=list
    )

    contradictions: list[str] = Field(
        default_factory=list
    )

    unresolved_questions: list[str] = Field(
        default_factory=list
    )