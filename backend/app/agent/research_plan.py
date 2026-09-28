"""Deterministic, in-memory planning and coverage primitives for MISSIONCTRL."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.research_executor import Evidence


ObjectiveStatus = Literal[
    "PENDING", "RESEARCHING", "PARTIALLY_COVERED", "SUFFICIENT", "UNRESOLVED"
]
ClaimStatus = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTESTED", "UNRESOLVED"]
SourceClassification = Literal[
    "PRIMARY", "HIGH_QUALITY_SECONDARY", "SPECIALIST", "GENERAL", "LOW_CONFIDENCE"
]


class ResearchObjective(BaseModel):
    id: str
    title: str
    description: str
    priority: int = Field(ge=1, le=5)
    research_questions: list[str] = Field(default_factory=list)
    status: ObjectiveStatus = "PENDING"
    evidence_count: int = 0
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unresolved_gaps: list[str] = Field(default_factory=list)
    supporting_sources: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    objective: str
    research_objectives: list[ResearchObjective] = Field(default_factory=list)
    cross_cutting_questions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    time_sensitive: bool = False


class ResearchSearchRecord(BaseModel):
    objective_id: str
    question: str
    query: str
    tool: str
    evidence_count: int
    useful_evidence_count: int


class Claim(BaseModel):
    text: str
    objective_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    support_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    contradictions: list[str] = Field(default_factory=list)
    status: ClaimStatus = "UNRESOLVED"


class CoverageMap(BaseModel):
    objectives: list[ResearchObjective] = Field(default_factory=list)
    overall_coverage: float = Field(default=0.0, ge=0.0, le=1.0)


class NextResearchAction(BaseModel):
    objective_id: str = ""
    reason: str
    missing: list[str] = Field(default_factory=list)
    next_question: str = ""


class MissionDecomposer:
    """Splits explicit mission dimensions without relying on an LLM call."""

    _VERB_BOUNDARY = re.compile(
        r"\b(?:compare|analy[sz]e|investigate|assess|evaluate|research|find|identify)\b",
        re.IGNORECASE,
    )

    def decompose(self, mission: str) -> ResearchPlan:
        text = " ".join(mission.split())
        chunks = [match.group().strip(" ,.;:") for match in self._VERB_BOUNDARY.finditer(text)]
        positions = [match.start() for match in self._VERB_BOUNDARY.finditer(text)]
        dimensions: list[str] = []
        for index, start in enumerate(positions):
            end = positions[index + 1] if index + 1 < len(positions) else len(text)
            segment = text[start:end].strip(" ,.;:")
            if segment:
                dimensions.extend(self._split_dimension(segment))

        # A single concise mission should remain a single objective. Only
        # distinct explicit dimensions receive separate research allocation.
        dimensions = list(dict.fromkeys(dimensions)) or [text]
        objectives = [self._objective(item, index) for index, item in enumerate(dimensions)]
        return ResearchPlan(
            objective=text,
            research_objectives=objectives,
            cross_cutting_questions=self._cross_cutting_questions(text),
            success_criteria=["Each high-priority mission dimension has direct, relevant evidence."],
            time_sensitive=bool(re.search(r"\b(latest|current|recent|today|this year|most recent)\b", text, re.I)),
        )

    def _split_dimension(self, segment: str) -> list[str]:
        # Split conjunctions only when they introduce independent noun phrases.
        pieces = re.split(r"\s+(?:and|,\s*and)\s+(?=(?:analy[sz]e|compare|[a-z]+\s+(?:market|product|supply|risk|constraint|offering)))", segment, flags=re.I)
        return [piece.strip() for piece in pieces if len(piece.strip()) > 8]

    def _objective(self, description: str, index: int) -> ResearchObjective:
        title = re.sub(r"^(?:investigate|compare|analy[sz]e|assess|evaluate|research|find|identify)\s+", "", description, flags=re.I).strip().capitalize()
        return ResearchObjective(
            id=f"objective-{index + 1}",
            title=title[:90],
            description=description,
            priority=max(1, 5 - index),
            research_questions=[
                f"What current, direct evidence addresses: {description}?",
                f"What independent source corroborates the key facts about: {description}?",
            ],
            unresolved_gaps=[title],
        )

    @staticmethod
    def _cross_cutting_questions(mission: str) -> list[str]:
        if re.search(r"\b(compare|versus|against)\b", mission, re.I):
            return ["What measurement period and methodology make the comparison valid?"]
        return []


class SourceQualityClassifier:
    """Classifies evidence using supplied content and metadata, not a site allowlist."""

    def classify(self, item: Evidence) -> SourceClassification:
        text = f"{item.title} {item.source} {item.snippet}".lower()
        if any(token in text for token in ("official", "filing", "documentation", "regulatory", "press release")):
            return "PRIMARY"
        if item.source_quality == "high":
            return "HIGH_QUALITY_SECONDARY"
        if any(token in text for token in ("research", "technical", "analysis", "industry")):
            return "SPECIALIST"
        if item.source_quality == "low" or not item.url:
            return "LOW_CONFIDENCE"
        return "GENERAL"


class CoverageAnalyzer:
    QUALITY_WEIGHT = {
        "PRIMARY": 1.0, "HIGH_QUALITY_SECONDARY": 0.85, "SPECIALIST": 0.7,
        "GENERAL": 0.45, "LOW_CONFIDENCE": 0.15,
    }

    def __init__(self, classifier: SourceQualityClassifier | None = None):
        self.classifier = classifier or SourceQualityClassifier()

    def update(self, objective: ResearchObjective, evidence: list[Evidence]) -> ResearchObjective:
        relevant = [item for item in evidence if item.relevant]
        sources = {item.source.strip().lower() for item in relevant if item.source.strip()}
        classifications = [self.classifier.classify(item) for item in relevant]
        for item, classification in zip(relevant, classifications):
            item.source_classification = classification
        quality = [self.QUALITY_WEIGHT[classification] for classification in classifications]
        directness = sum(item.relevance_score for item in relevant) / len(relevant) if relevant else 0.0
        quantity = min(len(relevant) / 3, 1.0)
        diversity = min(len(sources) / 2, 1.0)
        quality_score = sum(quality) / len(quality) if quality else 0.0
        contradiction_penalty = 0.25 if objective.contradictions else 0.0
        score = max(0.0, min(1.0, 0.35 * quantity + 0.25 * diversity + 0.25 * quality_score + 0.15 * directness - contradiction_penalty))
        objective.evidence_count = len(relevant)
        objective.coverage_score = round(score, 3)
        objective.supporting_sources = list(dict.fromkeys([item.url for item in relevant if item.url]))
        objective.status = "SUFFICIENT" if score >= 0.7 else "PARTIALLY_COVERED" if relevant else "UNRESOLVED"
        if objective.status == "SUFFICIENT":
            objective.unresolved_gaps = []
        return objective

    def map(self, objectives: list[ResearchObjective]) -> CoverageMap:
        weight = sum(objective.priority for objective in objectives) or 1
        overall = sum(objective.coverage_score * objective.priority for objective in objectives) / weight
        return CoverageMap(objectives=objectives, overall_coverage=round(overall, 3))


class ResearchPrioritizer:
    def select(self, objectives: list[ResearchObjective]) -> ResearchObjective | None:
        candidates = [item for item in objectives if item.status != "SUFFICIENT"]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.priority * (1 - item.coverage_score), item.priority))


class ClaimValidator:
    def validate(self, objective: ResearchObjective, evidence: list[Evidence]) -> Claim:
        supporting = [item for item in evidence if item.relevant]
        contradictions = list(dict.fromkeys(objective.contradictions + [
            str(item.metadata["contradiction"]) for item in supporting if item.metadata.get("contradiction")
        ]))
        strength = objective.coverage_score
        status: ClaimStatus = "CONTESTED" if contradictions else "SUPPORTED" if strength >= 0.7 else "PARTIALLY_SUPPORTED" if supporting else "UNRESOLVED"
        text = next((item.claim for item in supporting if item.claim), objective.title)
        return Claim(
            text=text or objective.title,
            objective_id=objective.id,
            evidence_ids=[item.url or item.title for item in supporting],
            support_strength=strength,
            contradictions=contradictions,
            status=status,
        )
