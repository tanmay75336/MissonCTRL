import asyncio

from app.agent.budget import SearchBudget
from app.agent.research_executor import ResearchExecutor
from app.agent.tool_router import ToolRouter


class FakeSerpApiTools:
    """
    Fake SerpApi client for testing.

    This test does NOT consume real SerpApi searches.
    """

    def web_search(self, query: str, location=None):
        return {
            "organic_results": [
                {
                    "position": 1,
                    "title": "Example Student Business Research",
                    "link": "https://example.com/student-business",
                    "snippet": "Example research result.",
                    "source": "Example Source",
                }
            ]
        }

    def news_search(self, query: str):
        return {
            "news_results": [
                {
                    "position": 1,
                    "title": "Example Student Business News",
                    "link": "https://example.com/news",
                    "snippet": "Example news result.",
                    "source": {
                        "name": "Example News"
                    },
                    "date": "2026-09-26",
                }
            ]
        }

    def maps_search(self, query: str):
        return {
            "local_results": [
                {
                    "title": "Example Student Cafe",
                    "link": "https://example.com/cafe",
                    "address": "Mumbai",
                    "rating": 4.5,
                    "reviews": 120,
                    "type": "Cafe",
                }
            ]
        }

    def trends_search(self, query: str, geo="IN"):
        return {
            "interest_over_time": {
                "timeline_data": []
            }
        }


async def main():

    router = ToolRouter()

    budget = SearchBudget(
        maximum=8
    )

    executor = ResearchExecutor(
        serpapi_tools=FakeSerpApiTools(),
        router=router,
        budget=budget,
    )

    questions = [
        "What unmet needs do students in Mumbai have?",
        "What are the latest student business announcements?",
        "Which student businesses are near Mumbai colleges?",
        "How has demand for student services changed over time?",
    ]

    print("\n" + "=" * 60)
    print("MISSIONCTRL — RESEARCH EXECUTOR TEST")
    print("=" * 60)

    for question in questions:

        result = await executor.research(
            question
        )

        print("\nQUESTION:")
        print(result.question)

        print("\nTOOL:")
        print(result.decision.tool)

        print("\nQUERY:")
        print(result.decision.query)

        print("\nSEARCH NUMBER:")
        print(result.search_number)

        print("\nREMAINING SEARCHES:")
        print(result.remaining_searches)

        print("\nEVIDENCE:")

        for item in result.evidence:

            print(
                f"\n  TITLE: {item.title}"
            )

            print(
                f"  URL: {item.url}"
            )

            print(
                f"  SOURCE: {item.source}"
            )

            print(
                f"  SNIPPET: {item.snippet}"
            )

        print("-" * 60)

    print("\nFINAL BUDGET:")
    print(budget.status())


if __name__ == "__main__":
    asyncio.run(main())