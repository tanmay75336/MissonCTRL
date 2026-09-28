from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.agent.research_executor import Evidence
from app.llm.base import LLMProvider


# ============================================================
# OUTPUT MODELS
# ============================================================


class SourceCitation(BaseModel):
    title: str
    url: str
    claim_supported: str


class FinalSynthesis(BaseModel):
    """
    Evidence-grounded final mission answer.
    """

    answer: str

    facts: list[str] = Field(
        default_factory=list
    )

    inferences: list[str] = Field(
        default_factory=list
    )

    unknowns: list[str] = Field(
        default_factory=list
    )

    unresolved_questions: list[str] = Field(
        default_factory=list
    )

    sources: list[SourceCitation] = Field(
        default_factory=list
    )


# ============================================================
# FINAL SYNTHESIZER
# ============================================================


class FinalSynthesizer:

    SYSTEM_PROMPT = """
You are the final evidence-grounded synthesis engine
of MISSIONCTRL.

Your job is to answer the ORIGINAL USER MISSION using
ONLY the research evidence supplied to you.

============================================================
CORE PRINCIPLE
============================================================

NEVER make the answer sound more certain than the evidence.

If the evidence does not establish something, say that
the evidence is insufficient.

MISSIONCTRL values accuracy and uncertainty over impressive
or speculative answers.

============================================================
FACTS
============================================================

A FACT must be directly supported by one or more supplied
evidence items.

Do NOT add information from your own knowledge.

Every important factual statement should be traceable to
the supplied evidence.

============================================================
INFERENCES
============================================================

An INFERENCE is a reasonable conclusion derived from
supported facts.

Clearly label it as an inference.

Do NOT turn an inference into a fact.

Example:

FACT:
Students report affordability problems.

INFERENCE:
A lower-cost service may be worth investigating.

Do NOT say:

"Students definitely want a low-cost service."

unless the evidence actually establishes that.

============================================================
UNKNOWN
============================================================

If the evidence does not establish something important,
put it under UNKNOWN.

Examples:

- actual startup cost is unknown
- willingness to pay is unknown
- market size is unknown
- strongest customer segment is unknown
- regulatory requirements are unknown

Do NOT fill these gaps with guesses.

============================================================
RECOMMENDATIONS
============================================================

Do NOT invent business ideas, products, companies,
financial projections, prices, revenue estimates, or
market opportunities merely because they sound reasonable.

If the evidence supports only a broad opportunity area,
say that.

If the evidence is insufficient to identify a specific
opportunity, explicitly say so.

============================================================
USER CONSTRAINTS
============================================================

Respect constraints stated in the ORIGINAL MISSION.

Do not change:

- budget
- location
- target audience
- timeline
- objective

Do not invent additional constraints.

============================================================
SOURCES
============================================================

Only cite URLs that appear in the supplied evidence.

Never create, guess, or modify URLs.

A source citation must explain exactly which claim
the source supports.

============================================================
FINAL ANSWER
============================================================

The answer must directly address the ORIGINAL MISSION.

If evidence is insufficient, the answer should say:

"The current evidence is insufficient to answer this
mission confidently."

Then explain what is known and what must be researched next.

Do NOT hide uncertainty.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{
  "answer": "...",

  "facts": [
    "..."
  ],

  "inferences": [
    "..."
  ],

  "unknowns": [
    "..."
  ],

  "unresolved_questions": [
    "..."
  ],

  "sources": [
    {
      "title": "...",
      "url": "...",
      "claim_supported": "..."
    }
  ]
}
"""

    def __init__(
        self,
        llm: LLMProvider,
    ):
        self.llm = llm

    async def synthesize(
        self,
        mission: str,
        evidence: list[Evidence],
        unresolved_questions: list[str],
        strategy_history: list[str],
        research_plan=None,
        claims=None,
        coverage=None,
    ) -> FinalSynthesis:

        # ----------------------------------------------------
        # Prepare evidence
        # ----------------------------------------------------

        evidence_payload = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            evidence_payload.append(
                {
                    "evidence_id": index,
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "source": item.source,
                    "published_at": item.published_at,
                    "claim": item.claim,
                    "relevance_score": (
                        item.relevance_score
                    ),
                    "source_quality": (
                        item.source_quality
                    ),
                    "critic_reasoning": (
                        item.critic_reasoning
                    ),
                    "tool": item.tool,
                    "query": item.query,
                }
            )

        # ----------------------------------------------------
        # Build synthesis request
        # ----------------------------------------------------

        user_prompt = f"""
ORIGINAL USER MISSION:
{mission}

STRATEGY HISTORY:
{json.dumps(
    strategy_history,
    ensure_ascii=False,
    indent=2,
)}

UNRESOLVED QUESTIONS:
{json.dumps(
    unresolved_questions,
    ensure_ascii=False,
    indent=2,
)}

SUPPLIED RESEARCH EVIDENCE:
{json.dumps(
    evidence_payload,
    ensure_ascii=False,
    indent=2,
)}

RESEARCH PLAN AND COVERAGE SUMMARY:
{json.dumps({
    "research_plan": research_plan.model_dump() if research_plan else {},
    "claims": [claim.model_dump() for claim in (claims or [])],
    "coverage": coverage.model_dump() if coverage else {},
}, ensure_ascii=False, indent=2)}

============================================================
YOUR TASK
============================================================

Answer the ORIGINAL USER MISSION.

First determine what the evidence actually establishes.

Then separate the result into:

FACTS:
Directly supported by the evidence.

INFERENCES:
Reasonable conclusions derived from those facts.

UNKNOWNS:
Important information that the evidence does not establish.

Do NOT invent specific business opportunities,
costs, demand levels, revenue, market sizes,
regulations, or customer preferences.

If the evidence is insufficient for a confident answer,
say so explicitly in the answer.

Every source you cite must come directly from the
supplied evidence.
"""

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        response = await self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                response
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Final synthesizer returned invalid JSON:\n"
                f"{response}"
            ) from error

        # ----------------------------------------------------
        # Validate structure
        # ----------------------------------------------------

        result = FinalSynthesis.model_validate(
            data
        )

        # ----------------------------------------------------
        # Source safety validation
        # ----------------------------------------------------

        evidence_urls = {
            item.url
            for item in evidence
            if item.url
        }

        for source in result.sources:

            if source.url not in evidence_urls:

                raise ValueError(
                    "Synthesizer returned a URL that was "
                    "not present in the supplied evidence: "
                    f"{source.url}"
                )

        # ----------------------------------------------------
        # Prevent empty answer
        # ----------------------------------------------------

        if not result.answer.strip():

            raise ValueError(
                "Synthesizer returned an empty answer."
            )

        return result
