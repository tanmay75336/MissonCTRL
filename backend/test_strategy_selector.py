import asyncio

from app.agent.strategy_engine import StrategyEngine
from app.agent.strategy_selector import StrategySelector
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

    # Generate strategies
    engine = StrategyEngine(llm=llm)

    strategy_set = await engine.generate(
        mission=MISSION,
        plan=PLAN,
    )

    print("\nGENERATED STRATEGIES:")

    for strategy in strategy_set.strategies:
        print(f"- {strategy.name}")

    # Select strategy
    selector = StrategySelector(llm=llm)

    selection = await selector.select(
        mission=MISSION,
        strategies=strategy_set.strategies,
    )

    print("\n" + "=" * 60)
    print("SELECTED STRATEGY")
    print("=" * 60)

    print(selection.selected_strategy)

    print("\nWHY:")
    print(selection.reason)

    print("\nEVALUATIONS:")

    for evaluation in selection.evaluations:
        print(f"\n{evaluation.strategy_name}")
        print(
            f"Information Value: {evaluation.information_value}"
        )
        print(
            f"Evidence Availability: "
            f"{evaluation.evidence_availability}"
        )
        print(
            f"Search Efficiency: "
            f"{evaluation.search_efficiency}"
        )
        print(
            f"Dead-End Risk: {evaluation.dead_end_risk}"
        )
        print(f"Reasoning: {evaluation.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())