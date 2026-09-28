import asyncio

from app.agent.budget import SearchBudget
from app.agent.evaluator import EvidenceEvaluator
from app.agent.research_executor import ResearchExecutor
from app.agent.research_loop import ResearchLoop
from app.tools.serpapi import SerpApiTools
from app.llm.factory import get_llm_provider


MISSION = (
    "I have ₹2 lakh and want to find "
    "a student-focused business opportunity in Mumbai."
)


async def main():

    print("\n" + "=" * 70)
    print("MISSIONCTRL — REAL AUTONOMOUS MISSION")
    print("=" * 70)

    print("\nMISSION:")
    print(MISSION)

    # --------------------------------
    # 1. LOAD LLM
    # --------------------------------

    print("\n[INIT] Loading Groq...")

    llm = get_llm_provider("groq")

    print("[INIT] Groq loaded.")

    # --------------------------------
    # 2. LOAD SERPAPI
    # --------------------------------

    print("\n[INIT] Loading SerpApi...")

    serpapi_tools = SerpApiTools()

    print("[INIT] SerpApi loaded.")

    # --------------------------------
    # 3. SEARCH BUDGET
    # --------------------------------

    budget = SearchBudget(
        maximum=4
    )

    print(
        f"[BUDGET] Maximum searches: "
        f"{budget.maximum}"
    )

    # --------------------------------
    # 4. RESEARCH EXECUTOR
    # --------------------------------
    #
    # ResearchExecutor now needs the LLM
    # because it generates focused search
    # queries before calling SerpApi.
    #

    executor = ResearchExecutor(
        serpapi_tools=serpapi_tools,
        llm=llm,
        budget=budget,
    )

    print(
        "[INIT] Research Executor loaded."
    )

    # --------------------------------
    # 5. EVIDENCE EVALUATOR
    # --------------------------------

    evaluator = EvidenceEvaluator()

    print(
        "[INIT] Evidence Evaluator loaded."
    )

    # --------------------------------
    # 6. AUTONOMOUS RESEARCH LOOP
    # --------------------------------

    loop = ResearchLoop(
        llm=llm,
        executor=executor,
        evaluator=evaluator,
    )

    print(
        "[INIT] Autonomous Research Loop loaded."
    )

    # --------------------------------
    # 7. RUN MISSION
    # --------------------------------

    print("\n" + "-" * 70)
    print("STARTING AUTONOMOUS RESEARCH")
    print("-" * 70)

    try:

        result = await loop.run(
            mission=MISSION
        )

    except Exception as error:

        print("\n" + "=" * 70)
        print("MISSION FAILED")
        print("=" * 70)

        print(
            f"\nERROR TYPE: "
            f"{type(error).__name__}"
        )

        print(
            f"ERROR: {error}"
        )

        raise

    # --------------------------------
    # 8. FINAL REPORT
    # --------------------------------

    print("\n" + "=" * 70)
    print("MISSIONCTRL — RESEARCH COMPLETE")
    print("=" * 70)

    # --------------------------------
    # OBJECTIVE
    # --------------------------------

    print("\nOBJECTIVE:")

    print(
        result.plan.objective
    )

    # --------------------------------
    # SEARCH STATISTICS
    # --------------------------------

    print("\nSEARCHES USED:")

    print(
        result.searches_used
    )

    print("\nSEARCHES REMAINING:")

    print(
        result.searches_remaining
    )

    print("\nITERATIONS:")

    print(
        result.iterations
    )

    # --------------------------------
    # COMPLETED QUESTIONS
    # --------------------------------

    print("\nCOMPLETED QUESTIONS:")

    if result.completed_questions:

        for question in result.completed_questions:

            print(
                f"- {question}"
            )

    else:

        print("- None")

    # --------------------------------
    # UNRESOLVED QUESTIONS
    # --------------------------------

    print("\nUNRESOLVED QUESTIONS:")

    if result.unresolved_questions:

        for question in result.unresolved_questions:

            print(
                f"- {question}"
            )

    else:

        print("- None")

    # --------------------------------
    # EVIDENCE
    # --------------------------------

    print("\nEVIDENCE COLLECTED:")

    if not result.evidence:

        print(
            "- No evidence collected."
        )

    else:

        for index, item in enumerate(
            result.evidence,
            start=1,
        ):

            print(
                "\n" + "-" * 60
            )

            print(
                f"EVIDENCE #{index}"
            )

            print(
                f"TITLE: {item.title}"
            )

            print(
                f"SOURCE: {item.source}"
            )

            print(
                f"TOOL: {item.tool}"
            )

            print(
                f"QUERY: {item.query}"
            )

            print(
                f"URL: {item.url}"
            )

            print(
                f"SNIPPET: {item.snippet}"
            )

    # --------------------------------
    # FINAL SEARCH BUDGET
    # --------------------------------

    print("\n" + "=" * 70)
    print("FINAL SEARCH BUDGET")
    print("=" * 70)

    print(
        budget.status()
    )

    print("\nMISSIONCTRL real mission finished.")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())