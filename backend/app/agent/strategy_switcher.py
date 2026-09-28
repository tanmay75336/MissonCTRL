from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.llm.base import LLMProvider


# ============================================================
# OUTPUT MODEL
# ============================================================


class StrategySwitchDecision(BaseModel):
    """
    Decision about whether MISSIONCTRL should switch
    from the current research strategy to another strategy.
    """

    should_switch: bool

    target_strategy: str = ""

    reason: str

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# STRATEGY SWITCHER
# ============================================================


class StrategySwitcher:
    """
    Determines whether the research agent should abandon
    the current strategy and continue with another strategy.

    The switcher is deliberately conservative.

    It should NOT switch when:

    - evidence is already strong
    - the current strategy has only been attempted once
    - there are no unresolved questions
    - no alternative strategies exist
    - the search budget is exhausted

    This prevents strategy thrashing and unnecessary searches.
    """

    SYSTEM_PROMPT = """
You are the strategy switching controller of MISSIONCTRL.

MISSIONCTRL is an autonomous research agent.

Your job is to decide whether the agent should continue
with its current research strategy or switch to another
available strategy.

============================================================
DECISION FACTORS
============================================================

Consider:

1. Evidence count
   How much evidence has been collected?

2. Evidence relevance
   Is the collected evidence actually relevant?

3. Unresolved questions
   Are important questions still unanswered?

4. Remaining search budget
   Can the agent afford additional research?

5. Attempts under the current strategy
   Has the current strategy already had a reasonable chance?

6. Available alternative strategies
   Is there a meaningful unused strategy to switch to?

============================================================
SWITCH WHEN
============================================================

Switch when:

- evidence is weak or insufficient,
- important questions remain unresolved,
- the current strategy has already been attempted,
- search budget remains,
- and another strategy could reasonably reduce
  the remaining uncertainty.

============================================================
DO NOT SWITCH WHEN
============================================================

Do NOT switch when:

- evidence is already strong,
- there are no unresolved questions,
- the current strategy has only just started,
- there is no meaningful alternative,
- or the remaining search budget is zero.

Avoid unnecessary strategy switching.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly:

{
  "should_switch": true,
  "target_strategy": "Strategy Name",
  "reason": "Explanation",
  "confidence": 0.0
}

If no switch is appropriate:

{
  "should_switch": false,
  "target_strategy": "",
  "reason": "Explanation",
  "confidence": 0.0
}
"""

    def __init__(
        self,
        llm: LLMProvider | None = None,
    ):
        self.llm = llm

    # ========================================================
    # DECIDE
    # ========================================================

    async def decide(
        self,
        current_strategy: str,
        available_strategies: list[str],
        evidence_count: int,
        average_relevance: float,
        unresolved_questions: int,
        remaining_budget: int,
        attempts_under_current_strategy: int = 1,
    ) -> StrategySwitchDecision:

        # ----------------------------------------------------
        # Basic safety checks
        # ----------------------------------------------------

        if not available_strategies:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "No alternative strategies are available."
                ),
                confidence=1.0,
            )

        if attempts_under_current_strategy < 2:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "The current strategy has not had "
                    "enough attempts to justify switching."
                ),
                confidence=0.95,
            )

        if unresolved_questions <= 0:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "There are no unresolved research "
                    "questions requiring another strategy."
                ),
                confidence=0.95,
            )

        if remaining_budget <= 0:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "The search budget is exhausted."
                ),
                confidence=1.0,
            )

        # ----------------------------------------------------
        # Strong evidence guard
        # ----------------------------------------------------

        if (
            evidence_count >= 3
            and average_relevance >= 0.75
        ):

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "The current strategy has produced "
                    "sufficiently relevant evidence, so "
                    "switching would add unnecessary search."
                ),
                confidence=0.9,
            )

        # ----------------------------------------------------
        # No LLM fallback
        # ----------------------------------------------------

        if self.llm is None:

            target = available_strategies[0]

            return StrategySwitchDecision(
                should_switch=True,
                target_strategy=target,
                reason=(
                    "Evidence remains insufficient and "
                    "unresolved questions remain. "
                    "Switching to an unused strategy."
                ),
                confidence=0.6,
            )

        # ----------------------------------------------------
        # Ask LLM to make the strategic decision
        # ----------------------------------------------------

        user_prompt = f"""
CURRENT STRATEGY:
{current_strategy}

AVAILABLE ALTERNATIVE STRATEGIES:
{json.dumps(
    available_strategies,
    ensure_ascii=False,
    indent=2,
)}

EVIDENCE COUNT:
{evidence_count}

AVERAGE RELEVANCE:
{average_relevance}

UNRESOLVED QUESTIONS:
{unresolved_questions}

REMAINING SEARCH BUDGET:
{remaining_budget}

ATTEMPTS UNDER CURRENT STRATEGY:
{attempts_under_current_strategy}

Determine whether MISSIONCTRL should switch strategies.

Do not switch merely because another strategy exists.

Switch only if the current strategy is producing
insufficient progress and another strategy is likely
to reduce important remaining uncertainty.
"""

        try:

            response = await self.llm.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

        except Exception as error:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "Strategy switch decision failed: "
                    f"{error}"
                ),
                confidence=0.0,
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = json.loads(response)

        except json.JSONDecodeError:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "Strategy switcher returned invalid JSON."
                ),
                confidence=0.0,
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        try:

            decision = (
                StrategySwitchDecision.model_validate(
                    data
                )
            )

        except Exception as error:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "Invalid strategy switch decision: "
                    f"{error}"
                ),
                confidence=0.0,
            )

        # ----------------------------------------------------
        # Validate target strategy
        # ----------------------------------------------------

        if not decision.should_switch:

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=decision.reason,
                confidence=decision.confidence,
            )

        if (
            not decision.target_strategy
            or decision.target_strategy
            not in available_strategies
        ):

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "The proposed target strategy is "
                    "not an available alternative."
                ),
                confidence=0.0,
            )

        # ----------------------------------------------------
        # Prevent switching to current strategy
        # ----------------------------------------------------

        if (
            decision.target_strategy
            == current_strategy
        ):

            return StrategySwitchDecision(
                should_switch=False,
                target_strategy="",
                reason=(
                    "The proposed strategy is the same "
                    "as the current strategy."
                ),
                confidence=0.0,
            )

        return decision