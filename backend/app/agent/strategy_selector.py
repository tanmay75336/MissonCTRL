from __future__ import annotations

import json
from pydantic import BaseModel, Field

from app.llm.base import LLMProvider
from app.agent.strategy_engine import ResearchStrategy


class StrategyEvaluation(BaseModel):
    strategy_name: str

    information_value: float = Field(ge=0, le=1)
    evidence_availability: float = Field(ge=0, le=1)
    search_efficiency: float = Field(ge=0, le=1)
    dead_end_risk: float = Field(ge=0, le=1)

    reasoning: str


class StrategySelection(BaseModel):
    selected_strategy: str
    evaluations: list[StrategyEvaluation]
    reason: str


class StrategySelector:

    SYSTEM_PROMPT = """
You are the strategy selector for MISSIONCTRL.

MISSIONCTRL has generated multiple possible research strategies.
Your job is to determine which strategy should receive the
initial research budget.

Evaluate every strategy using:

1. information_value
   How much important uncertainty could this strategy reduce?

2. evidence_availability
   How likely is it that reliable evidence can be found?

3. search_efficiency
   How much useful information can be obtained per search?

4. dead_end_risk
   How likely is the strategy to produce weak, irrelevant,
   or difficult-to-verify evidence?

Be critical. Do not select a strategy simply because it sounds good.

Choose exactly ONE strategy.

Return ONLY valid JSON:

{
  "selected_strategy": "...",
  "evaluations": [
    {
      "strategy_name": "...",
      "information_value": 0.0,
      "evidence_availability": 0.0,
      "search_efficiency": 0.0,
      "dead_end_risk": 0.0,
      "reasoning": "..."
    }
  ],
  "reason": "..."
}
"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def select(
        self,
        mission: str,
        strategies: list[ResearchStrategy],
    ) -> StrategySelection:

        strategies_text = json.dumps(
            [strategy.model_dump() for strategy in strategies],
            indent=2,
            ensure_ascii=False,
        )

        user_prompt = f"""
USER MISSION:
{mission}

AVAILABLE STRATEGIES:
{strategies_text}

Evaluate all strategies and select the one that should
receive the initial research budget.
"""

        response = await self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Strategy selector returned invalid JSON: {response}"
            ) from error

        result = StrategySelection.model_validate(data)

        valid_names = {strategy.name for strategy in strategies}

        if result.selected_strategy not in valid_names:
            raise ValueError(
                f"Selected strategy '{result.selected_strategy}' "
                "does not exist."
            )

        return result