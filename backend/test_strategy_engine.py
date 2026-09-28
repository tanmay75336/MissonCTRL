import asyncio

from app.agent.strategy_engine import StrategyEngine
from app.llm.factory import get_llm_provider


MISSION = """
I have ₹2 lakh and want to find a student-focused
business opportunity in Mumbai.
"""


PLAN = """
Objective:
Identify a viable student-focused business opportunity
in Mumbai within a ₹2 lakh budget.

Constraints:
- Budget: ₹2 lakh
- Location: Mumbai
- Target: Students

Unknowns:
- Student segment
- Unmet needs
- Competition
- Startup costs
- Regulations
- Demand
"""


async def main():
    llm = get_llm_provider("groq")

    engine = StrategyEngine(llm=llm)

    result = await engine.generate(
        mission=MISSION,
        plan=PLAN,
    )

    for i, strategy in enumerate(result.strategies, 1):
        print(f"\n{'=' * 60}")
        print(f"STRATEGY {i}: {strategy.name}")
        print(f"{'=' * 60}")

        print(f"\nObjective:\n{strategy.objective}")
        print(f"\nApproach:\n{strategy.approach}")

        print("\nResearch Questions:")
        for question in strategy.research_questions:
            print(f"- {question}")

        print("\nStrengths:")
        for item in strategy.strengths:
            print(f"- {item}")

        print("\nWeaknesses:")
        for item in strategy.weaknesses:
            print(f"- {item}")


if __name__ == "__main__":
    asyncio.run(main())