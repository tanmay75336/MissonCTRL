from typing import Literal

from pydantic import BaseModel, Field

from app.agent.research_executor import Evidence


class EvaluationResult(BaseModel):
    """
    Evaluation of the evidence collected for a research question.
    """

    question: str

    evidence_count: int

    unique_sources: int

    has_relevant_evidence: bool

    has_source_diversity: bool

    sufficient: bool

    missing_information: list[str] = Field(
        default_factory=list
    )

    follow_up_questions: list[str] = Field(
        default_factory=list
    )

    reason: str

    relevant_evidence_count: int = 0

    directly_answers_question: bool = False

    contradictions: list[str] = Field(
        default_factory=list
    )

    actionable_gaps: list[str] = Field(
        default_factory=list
    )


class NextActionDecision(BaseModel):
    """An explicit, bounded decision about what research does next."""

    status: Literal["SUFFICIENT", "INSUFFICIENT"]
    reason: str
    missing_information: list[str] = Field(default_factory=list)
    next_action: Literal["SYNTHESIZE", "SEARCH", "STOP"]
    next_question: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceEvaluator:
    """
    Evaluates whether collected evidence is sufficient
    to answer a research question.
    """

    MIN_EVIDENCE = 3
    MIN_SOURCES = 2

    def evaluate(
        self,
        question: str,
        evidence: list[Evidence],
        contradictions: list[str] | None = None,
    ) -> EvaluationResult:

        evidence_count = len(evidence)

        # --------------------------------
        # UNIQUE SOURCES
        # --------------------------------

        sources = set()

        for item in evidence:

            source = (
                item.source.strip().lower()
                if item.source
                else ""
            )

            if source:
                sources.add(source)

            elif item.url:
                sources.add(
                    item.url.split("/")[2]
                    if "://" in item.url
                    else item.url
                )

        unique_sources = len(sources)

        # --------------------------------
        # RELEVANCE CHECK
        # --------------------------------

        relevant_evidence = self._find_relevant_evidence(
            question,
            evidence,
        )

        has_relevant_evidence = len(relevant_evidence) > 0

        directly_answers_question = any(
            (item.snippet.strip() or (item.claim or "").strip())
            for item in relevant_evidence
        )

        detected_contradictions = list(contradictions or [])
        detected_contradictions.extend(
            self._find_contradictions(evidence)
        )
        detected_contradictions = list(
            dict.fromkeys(detected_contradictions)
        )

        actionable_gaps = self._identify_actionable_gaps(
            question=question,
            evidence=evidence,
            contradictions=detected_contradictions,
        )

        # --------------------------------
        # SOURCE DIVERSITY
        # --------------------------------

        has_source_diversity = (
            unique_sources >= self.MIN_SOURCES
        )

        # --------------------------------
        # IDENTIFY MISSING INFORMATION
        # --------------------------------

        missing_information = []

        if evidence_count == 0:
            missing_information.append(
                "No evidence was returned."
            )

        if not has_relevant_evidence:
            missing_information.append(
                "No clearly relevant evidence was found."
            )

        if evidence_count < self.MIN_EVIDENCE:
            missing_information.append(
                "More evidence is needed."
            )

        if not has_source_diversity:
            missing_information.append(
                "Evidence comes from too few independent sources."
            )

        if not directly_answers_question:
            missing_information.append(
                "The available evidence does not directly answer the question."
            )

        if detected_contradictions:
            missing_information.append(
                "Available evidence contains unresolved contradictions."
            )

        # Diagnostic items above explain why evidence is insufficient. These
        # concrete gaps are what the autonomous loop should actually search.
        for gap in actionable_gaps:
            if gap not in missing_information:
                missing_information.append(gap)

        # --------------------------------
        # FOLLOW-UP QUESTIONS
        # --------------------------------

        follow_up_questions = []

        if evidence_count == 0:

            follow_up_questions.append(
                question
            )

        elif evidence_count < self.MIN_EVIDENCE:

            follow_up_questions.append(
                self._create_follow_up_question(
                    question
                )
            )

        elif not has_source_diversity:

            follow_up_questions.append(
                self._create_source_validation_question(
                    question
                )
            )

        # --------------------------------
        # FINAL DECISION
        # --------------------------------

        authoritative_single_source = (
            self._is_narrow_factual_question(question)
            and len(relevant_evidence) == 1
            and relevant_evidence[0].source_quality == "high"
            and directly_answers_question
        )

        sufficient = (
            not detected_contradictions
            and directly_answers_question
            and has_relevant_evidence
            and (
                authoritative_single_source
                or (
                    len(relevant_evidence) >= self.MIN_EVIDENCE
                    and has_source_diversity
                )
            )
        )

        if authoritative_single_source:
            missing_information = [
                item
                for item in missing_information
                if item not in {
                    "More evidence is needed.",
                    "Evidence comes from too few independent sources.",
                }
            ]

        if sufficient:

            reason = (
                "The research question has enough "
                "relevant evidence from multiple sources."
            )

        else:

            reason = (
                "More research is required because "
                "the current evidence does not satisfy "
                "the minimum coverage requirements."
            )

        return EvaluationResult(
            question=question,
            evidence_count=evidence_count,
            unique_sources=unique_sources,
            has_relevant_evidence=has_relevant_evidence,
            has_source_diversity=has_source_diversity,
            sufficient=sufficient,
            missing_information=missing_information,
            follow_up_questions=follow_up_questions,
            reason=reason,
            relevant_evidence_count=len(relevant_evidence),
            directly_answers_question=directly_answers_question,
            contradictions=detected_contradictions,
            actionable_gaps=actionable_gaps,
        )

    def decide_next_action(
        self,
        question: str,
        evidence: list[Evidence],
        remaining_budget: int,
        contradictions: list[str] | None = None,
    ) -> NextActionDecision:
        """Choose synthesis, a targeted search, or a safe stop.

        This is deliberately deterministic: the critic has already supplied
        relevance signals, while this layer keeps budget and stop behavior
        predictable and testable.
        """

        evaluation = self.evaluate(
            question=question,
            evidence=evidence,
            contradictions=contradictions,
        )

        if evaluation.sufficient:
            return NextActionDecision(
                status="SUFFICIENT",
                reason=evaluation.reason,
                next_action="SYNTHESIZE",
                confidence=0.9,
            )

        if remaining_budget <= 0:
            return NextActionDecision(
                status="INSUFFICIENT",
                reason=(
                    "The available evidence remains insufficient and the "
                    "search budget is exhausted."
                ),
                missing_information=evaluation.missing_information,
                next_action="STOP",
                confidence=1.0,
            )

        next_question = self.create_targeted_follow_up(
            mission=question,
            evaluation=evaluation,
        )

        return NextActionDecision(
            status="INSUFFICIENT",
            reason=evaluation.reason,
            missing_information=evaluation.missing_information,
            next_action="SEARCH",
            next_question=next_question,
            confidence=0.8,
        )

    def _find_relevant_evidence(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> list[Evidence]:

        question_words = self._keywords(
            question
        )

        relevant = []

        for item in evidence:

            if not item.relevant:
                continue

            text = " ".join(
                [
                    item.title,
                    item.snippet,
                ]
            ).lower()

            matches = sum(
                1
                for word in question_words
                if word in text
            )

            if matches >= 1:
                relevant.append(item)

        return relevant

    @staticmethod
    def _is_narrow_factual_question(question: str) -> bool:
        broad_terms = {
            "compare", "analysis", "analyze", "all", "best",
            "market", "opportunities", "needs", "problems",
        }
        words = set(question.lower().split())
        return (
            question.strip().endswith("?")
            and len(words) <= 14
            and not (words & broad_terms)
        )

    @staticmethod
    def _find_contradictions(
        evidence: list[Evidence],
    ) -> list[str]:
        contradictions: list[str] = []

        for item in evidence:
            marker = item.metadata.get("contradiction")
            if isinstance(marker, str) and marker.strip():
                contradictions.append(marker.strip())
            elif marker is True:
                contradictions.append(
                    f"Conflicting evidence reported by {item.source or item.title}."
                )

        return contradictions

    def create_targeted_follow_up(
        self,
        mission: str,
        evaluation: EvaluationResult,
        excluded_questions: set[str] | None = None,
    ) -> str:
        """Turn one concrete evidence gap into a materially focused search."""

        excluded = excluded_questions or set()
        gaps = evaluation.actionable_gaps or self._identify_actionable_gaps(
            question=mission,
            evidence=[],
            contradictions=evaluation.contradictions,
        )

        for gap in gaps:
            candidate = self._question_for_gap(gap, mission)
            if (
                candidate
                and self._normalize_text(candidate) not in excluded
            ):
                return candidate
        return ""

    def _identify_actionable_gaps(
        self,
        question: str,
        evidence: list[Evidence],
        contradictions: list[str],
    ) -> list[str]:
        """Infer small researchable categories from explicit mission wording.

        This intentionally does not manufacture facts. It extracts only
        categories that the user explicitly asked to establish.
        """

        text = question.lower()
        gaps: list[str] = []
        categories = [
            (
                ("market share", "market positioning", "sales volume", "revenue"),
                "comparative market share and sales data",
            ),
            (
                ("product offering", "product lineup", "products", "mi series"),
                "current product lineup comparison",
            ),
            (
                ("supply chain", "fabrication", "packaging", "semiconductor", "memory"),
                "semiconductor fabrication, packaging, and memory supply constraints",
            ),
            (
                ("judging criteria", "criteria"),
                "official judging criteria",
            ),
            (
                ("submission requirements", "submission"),
                "official submission requirements",
            ),
            (
                ("rules", "constraints", "requirements"),
                "important official rules and constraints",
            ),
        ]
        for keywords, gap in categories:
            if any(keyword in text for keyword in keywords):
                gaps.append(gap)

        if contradictions:
            gaps.insert(0, "authoritative resolution of conflicting information")

        # A narrowly factual question still has a specific, usable subject.
        if not gaps:
            keywords = sorted(self._keywords(question))
            if keywords:
                gaps.append("direct evidence about " + " ".join(keywords[:5]))

        return list(dict.fromkeys(gaps))

    def _question_for_gap(self, gap: str, mission: str) -> str:
        subject = self._mission_subject(mission)
        templates = {
            "comparative market share and sales data": (
                "What recent independent data compares market share and sales "
                f"for {subject}?"
            ),
            "current product lineup comparison": (
                f"What are the current product offerings for {subject}, and how do they compare?"
            ),
            "semiconductor fabrication, packaging, and memory supply constraints": (
                "What current semiconductor fabrication, packaging, and memory "
                f"supply constraints affect {subject}?"
            ),
            "official judging criteria": (
                "Which independent primary source defines the official judging "
                f"criteria for {subject}?"
            ),
            "official submission requirements": (
                "Which official source defines the submission requirements "
                f"for {subject}?"
            ),
            "important official rules and constraints": (
                "Which official documentation states the rules and constraints "
                f"for {subject}?"
            ),
            "authoritative resolution of conflicting information": (
                f"What authoritative source resolves conflicting information about {subject}?"
            ),
        }
        return templates.get(gap, f"Find direct evidence about {gap} for {subject}.")

    @staticmethod
    def _mission_subject(mission: str) -> str:
        """Use the mission's named terms without inventing new entities."""

        excluded = {"what", "when", "where", "which", "how", "find", "compare"}
        words = [
            word.strip(".,?!:;()[]{}")
            for word in mission.split()
            if (word[:1].isupper() or word.isupper())
            and word.lower().strip(".,?!:;()[]{}") not in excluded
        ]
        subject = " ".join(dict.fromkeys(words))
        return subject or "the entities and topic named in the mission"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(
            "".join(character if character.isalnum() else " " for character in text.lower()).split()
        )

    def _keywords(
        self,
        text: str,
    ) -> set[str]:

        stop_words = {
            "what",
            "which",
            "where",
            "when",
            "why",
            "how",
            "are",
            "is",
            "the",
            "a",
            "an",
            "in",
            "on",
            "for",
            "of",
            "to",
            "and",
            "or",
            "with",
            "do",
            "does",
            "have",
            "has",
            "been",
        }

        words = text.lower().split()

        return {
            word.strip(".,?!:;()[]{}")
            for word in words
            if len(word) >= 4
            and word not in stop_words
        }

    def _create_follow_up_question(
        self,
        question: str,
    ) -> str:

        return f"Find direct evidence about the specific unresolved aspect of: {question}"

    def _create_source_validation_question(
        self,
        question: str,
    ) -> str:

        return f"Find an independent source for the specific unresolved aspect of: {question}"
