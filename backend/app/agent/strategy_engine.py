from __future__ import annotations

import json
from pydantic import BaseModel, Field

from app.llm.base import LLMProvider


class ResearchStrategy(BaseModel):
    name: str
    objective: str
    approach: str
    research_questions: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class StrategySet(BaseModel):
    strategies: list[ResearchStrategy]


class StrategyEngine:
    SYSTEM_PROMPT = """
You are the strategy engine of MISSIONCTRL, an autonomous
research and decision-making agent.

Your job is to create multiple genuinely different research
strategies for solving a user's mission.

A strategy is NOT simply a different wording of the same search.

Each strategy should:
- investigate the mission from a different angle
- have a clear objective
- contain focused research questions
- explain its strengths and weaknesses
- remain grounded in the user's actual constraints
- never invent facts, numbers, dates, budgets, or requirements

Generate 2 to 3 strategies.

Return ONLY valid JSON in this format:

{
  "strategies": [
    {
      "name": "...",
      "objective": "...",
      "approach": "...",
      "research_questions": ["...", "..."],
      "strengths": ["..."],
      "weaknesses": ["..."]
    }
  ]
}
"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def generate(
        self,
        mission: str,
        plan: str,
    ) -> StrategySet:

        user_prompt = f"""
USER MISSION:
{mission}

MISSION PLAN:
{plan}

Create 2-3 substantially different research strategies
for solving this mission.
"""

        response = await self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Strategy engine returned invalid JSON: {response}"
            ) from error

        result = StrategySet.model_validate(data)

        if len(result.strategies) < 2:
            raise ValueError(
                "Strategy engine must generate at least 2 strategies."
            )

        return result