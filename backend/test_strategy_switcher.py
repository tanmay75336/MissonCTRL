import asyncio

from app.agent.strategy_switcher import StrategySwitcher
from app.llm.groq import GroqProvider


async def main():

    print("=" * 70)
    print("MISSIONCTRL — STRATEGY SWITCHER TEST")
    print("=" * 70)

    llm = GroqProvider()

    switcher = StrategySwitcher(
        llm=llm
    )

    # ========================================================
    # TEST 1
    # Strong evidence
    # ========================================================

    print()
    print("TEST 1 — STRONG EVIDENCE")

    result = await switcher.decide(
        current_strategy="Demand First",
        available_strategies=[
            "Market Gap First",
            "Cost Feasibility First",
        ],
        evidence_count=5,
        average_relevance=0.92,
        unresolved_questions=1,
        remaining_budget=3,
        attempts_under_current_strategy=2,
    )

    print(result)

    # ========================================================
    # TEST 2
    # Zero evidence
    # ========================================================

    print()
    print("TEST 2 — ZERO EVIDENCE")

    result = await switcher.decide(
        current_strategy="Demand First",
        available_strategies=[
            "Market Gap First",
            "Cost Feasibility First",
        ],
        evidence_count=0,
        average_relevance=0.0,
        unresolved_questions=3,
        remaining_budget=3,
        attempts_under_current_strategy=2,
    )

    print(result)

    # ========================================================
    # TEST 3
    # Weak evidence + low budget
    # ========================================================

    print()
    print("TEST 3 — WEAK EVIDENCE + LOW BUDGET")

    result = await switcher.decide(
        current_strategy="Market Gap First",
        available_strategies=[
            "Demand First",
            "Cost Feasibility First",
        ],
        evidence_count=1,
        average_relevance=0.30,
        unresolved_questions=2,
        remaining_budget=1,
        attempts_under_current_strategy=2,
    )

    print(result)

    # ========================================================
    # TEST 4
    # Attempt 1 — MUST NOT SWITCH
    # ========================================================

    print()
    print("TEST 4 — FIRST ATTEMPT")

    result = await switcher.decide(
        current_strategy="Demand First",
        available_strategies=[
            "Market Gap First",
            "Cost Feasibility First",
        ],
        evidence_count=0,
        average_relevance=0.0,
        unresolved_questions=3,
        remaining_budget=3,
        attempts_under_current_strategy=1,
    )

    print(result)

    assert result.should_switch is False

    print()
    print("PASS — first attempt did not switch.")

    print()
    print("=" * 70)
    print("ALL STRATEGY SWITCHER TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())