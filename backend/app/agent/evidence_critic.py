from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.llm.base import LLMProvider


if TYPE_CHECKING:
    from app.agent.research_executor import Evidence


# ============================================================
# EVIDENCE ASSESSMENT
# ============================================================

class EvidenceAssessment(BaseModel):

    index: int = 0

    relevant: bool = False

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    claim: str | None = None

    source_quality: str = "unknown"

    reasoning: str = ""


# ============================================================
# BATCH RESPONSE
# ============================================================

class BatchEvidenceAssessment(BaseModel):

    assessments: list[EvidenceAssessment] = Field(
        default_factory=list
    )


# ============================================================
# EVIDENCE CRITIC
# ============================================================

class EvidenceCritic:
    """
    Evaluates research evidence using an LLM.

    Supports:

        assess()
            → evaluates one result

        assess_many()
            → evaluates multiple results in ONE LLM call

    The autonomous research pipeline uses assess_many()
    to reduce LLM calls and latency.
    """

    SYSTEM_PROMPT = """
You are the evidence-quality critic for an autonomous
research agent.

Your job is to determine whether search results provide
useful evidence for a specific research question.

For every result determine:

1. relevant
   - Does this result directly help answer the research question?

2. relevance_score
   - 0.0 = completely irrelevant
   - 0.25 = weak connection
   - 0.50 = somewhat useful
   - 0.75 = strongly useful
   - 1.0 = directly useful evidence

3. claim
   - Extract the useful factual claim supported by the result.
   - Do not invent information.

4. source_quality
   - high
   - medium
   - low
   - unknown

5. reasoning
   - Briefly explain why the result is or is not useful.

IMPORTANT RULES:

- Do NOT treat keyword overlap as relevance.
- Do NOT invent facts.
- Do NOT assume that a result is useful merely because
  it contains words such as "Mumbai", "student", or "business".
- Reject unrelated results.
- Generic business-idea articles are NOT automatically
  evidence of student demand.
- A result about one institution does NOT automatically
  represent all Mumbai students.
- Distinguish relevance from source quality.
- Use only information contained in the supplied result.
- Preserve the original result index.
- Be conservative when evidence is weak.

Return ONLY valid JSON.

For batch evaluation use:

{
  "assessments": [
    {
      "index": 0,
      "relevant": true,
      "relevance_score": 0.85,
      "claim": "Students reported concerns about ...",
      "source_quality": "high",
      "reasoning": "The article directly addresses the research question."
    }
  ]
}
"""

    def __init__(
        self,
        llm: LLMProvider,
    ):
        self.llm = llm

    # ========================================================
    # SINGLE EVIDENCE — BACKWARD COMPATIBILITY
    # ========================================================

    async def assess(
        self,
        question: str,
        evidence: Evidence,
    ) -> EvidenceAssessment:

        """
        Evaluate one evidence item.

        This method is kept for existing tests and
        single-result use cases.

        Internally it uses assess_many() so there is
        only one evaluation implementation.
        """

        results = await self.assess_many(
            question=question,
            evidence_items=[evidence],
        )

        if not results:

            return EvidenceAssessment(
                index=0,
                relevant=False,
                relevance_score=0.0,
                claim="",
                source_quality="unknown",
                reasoning=(
                    "The evidence critic returned no assessment."
                ),
            )

        result = results[0]

        # Ensure single-result callers always receive index 0.

        result.index = 0

        return result

    # ========================================================
    # BATCH EVIDENCE
    # ========================================================

    async def assess_many(
        self,
        question: str,
        evidence_items: list[Evidence],
    ) -> list[EvidenceAssessment]:

        """
        Evaluate multiple evidence items in ONE LLM call.

        This is the method used by ResearchExecutor.
        """

        if not evidence_items:

            return []

        # ----------------------------------------------------
        # BUILD SEARCH RESULTS FOR LLM
        # ----------------------------------------------------

        results_text: list[str] = []

        for index, evidence in enumerate(
            evidence_items
        ):

            results_text.append(
                f"""
RESULT INDEX: {index}

TITLE:
{evidence.title}

SOURCE:
{evidence.source}

URL:
{evidence.url}

SNIPPET:
{evidence.snippet}

SEARCH QUERY:
{evidence.query}
"""
            )

        # ----------------------------------------------------
        # USER PROMPT
        # ----------------------------------------------------

        user_prompt = f"""
RESEARCH QUESTION:

{question}

SEARCH RESULTS:

{"".join(results_text)}
"""

        # ----------------------------------------------------
        # LLM CALL
        # ----------------------------------------------------

        response = await self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                response
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Evidence critic returned invalid JSON:\n"
                f"{response}"
            ) from error

        # ----------------------------------------------------
        # VALIDATE RESPONSE
        # ----------------------------------------------------

        try:

            result = (
                BatchEvidenceAssessment.model_validate(
                    data
                )
            )

        except Exception as error:

            raise ValueError(
                "Evidence critic returned an invalid "
                "assessment structure:\n"
                f"{data}"
            ) from error

        # ----------------------------------------------------
        # FILTER INVALID INDEXES
        # ----------------------------------------------------

        valid_results: list[
            EvidenceAssessment
        ] = []

        for assessment in result.assessments:

            if (
                0
                <= assessment.index
                < len(evidence_items)
            ):

                valid_results.append(
                    assessment
                )

        return valid_results