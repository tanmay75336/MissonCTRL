import asyncio

from app.agent.budget import SearchBudget
from app.agent.research_executor import ResearchExecutor
from app.llm.factory import get_llm_provider


MISSION = (
    "I have ₹2 lakh and want to find "
    "a student-focused business opportunity in Mumbai."
)


class FakeSerpApiTools:
    """
    Fake SerpApi implementation.

    No real SerpApi request is made.
    """

    def web_search(self, query: str, location=None):

        print(
            f"\n[FAKE SERPAPI] Query received:"
        )

        print(query)

        return {
            "organic_results": [

                {
                    "title": (
                        "IIT Bombay students seek better "
                        "student welfare and mental health support"
                    ),
                    "link": (
                        "https://example.com/iit-bombay-welfare"
                    ),
                    "snippet": (
                        "Students at IIT Bombay have raised "
                        "concerns about student welfare and "
                        "mental health support."
                    ),
                    "source": "Hindustan Times",
                    "position": 1,
                },

                {
                    "title": (
                        "Caliber Mining IPO 2026 Complete Review"
                    ),
                    "link": (
                        "https://example.com/caliber-mining"
                    ),
                    "snippet": (
                        "Complete review of the Caliber Mining "
                        "IPO including price band and subscription."
                    ),
                    "source": "India Infoline",
                    "position": 2,
                },

                {
                    "title": (
                        "Top Franchise Businesses in India"
                    ),
                    "link": (
                        "https://example.com/franchise-business"
                    ),
                    "snippet": (
                        "A list of franchise business opportunities "
                        "that can be started in India."
                    ),
                    "source": "SugarMint",
                    "position": 3,
                },
            ]
        }

    def news_search(self, query: str):

        return {
            "news_results": []
        }

    def maps_search(self, query: str):

        return {
            "local_results": []
        }

    def trends_search(
        self,
        query: str,
        geo: str = "IN",
    ):

        return {}


async def main():

    print("\n" + "=" * 70)
    print("MISSIONCTRL — RESEARCH EXECUTOR + CRITIC TEST")
    print("=" * 70)

    print("\nMISSION:")
    print(MISSION)

    # --------------------------------
    # 1. LOAD GROQ
    # --------------------------------

    print("\n[INIT] Loading Groq...")

    llm = get_llm_provider("groq")

    print("[INIT] Groq loaded.")

    # --------------------------------
    # 2. FAKE SERPAPI
    # --------------------------------

    print("\n[INIT] Loading Fake SerpApi...")

    serpapi_tools = FakeSerpApiTools()

    print(
        "[INIT] Fake SerpApi loaded."
    )

    # --------------------------------
    # 3. SEARCH BUDGET
    # --------------------------------

    budget = SearchBudget(
        maximum=1
    )

    # --------------------------------
    # 4. RESEARCH EXECUTOR
    # --------------------------------

    executor = ResearchExecutor(
        serpapi_tools=serpapi_tools,
        llm=llm,
        budget=budget,
    )

    # --------------------------------
    # 5. RUN RESEARCH
    # --------------------------------

    question = (
        "What unmet needs do students in "
        "Mumbai currently experience?"
    )

    print("\n" + "-" * 70)
    print("STARTING EXECUTOR TEST")
    print("-" * 70)

    result = await executor.research(
        question=question,
        context=MISSION,
    )

    # --------------------------------
    # 6. DISPLAY RESULTS
    # --------------------------------

    print("\n" + "=" * 70)
    print("RESEARCH RESULT")
    print("=" * 70)

    print("\nORIGINAL QUESTION:")
    print(result.question)

    print("\nSELECTED TOOL:")
    print(result.decision.tool)

    print("\nGENERATED QUERY:")
    print(result.decision.query)

    print("\nRAW EVIDENCE COUNT:")
    print(result.raw_evidence_count)

    print("\nFILTERED EVIDENCE COUNT:")
    print(result.filtered_evidence_count)

    print("\nFILTERED EVIDENCE:")

    if not result.evidence:

        print("- No relevant evidence.")

    else:

        for index, item in enumerate(
            result.evidence,
            start=1,
        ):

            print("\n" + "-" * 60)

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
                f"RELEVANT: {item.relevant}"
            )

            print(
                f"SCORE: {item.relevance_score}"
            )

            print(
                f"CLAIM: {item.claim}"
            )

            print(
                f"SOURCE QUALITY: {item.source_quality}"
            )

            print(
                f"REASONING: {item.critic_reasoning}"
            )

    # --------------------------------
    # 7. BUDGET
    # --------------------------------

    print("\n" + "=" * 70)
    print("BUDGET")
    print("=" * 70)

    print(
        budget.status()
    )

    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())