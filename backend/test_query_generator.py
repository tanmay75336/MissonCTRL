import asyncio

from app.agent.query_generator import QueryGenerator
from app.llm.factory import get_llm_provider


MISSION = (
    "I have ₹2 lakh and want to find "
    "a student-focused business opportunity in Mumbai."
)


async def main():

    print("\n" + "=" * 60)
    print("MISSIONCTRL — QUERY GENERATOR TEST")
    print("=" * 60)

    llm = get_llm_provider("groq")

    generator = QueryGenerator(
        llm=llm
    )

    questions = [
        (
            "What unmet needs do students in Mumbai "
            "currently experience?",
            "web_search",
        ),
        (
            "What existing products or services "
            "already address those needs?",
            "web_search",
        ),
        (
            "What business models could be launched "
            "with an initial investment of ₹2 lakh?",
            "web_search",
        ),
        (
            "Which student-focused businesses are "
            "near Mumbai colleges?",
            "maps_search",
        ),
        (
            "What are the latest developments in the "
            "Mumbai student business market?",
            "news_search",
        ),
    ]

    for index, (question, tool) in enumerate(
        questions,
        start=1,
    ):

        print(
            f"\nQUESTION #{index}"
        )

        print(
            f"Research question:\n{question}"
        )

        print(
            f"Tool: {tool}"
        )

        result = await generator.generate(
            mission=MISSION,
            question=question,
            tool=tool,
        )

        print(
            f"\nGENERATED QUERY:"
        )

        print(
            result.query
        )

        print(
            f"\nREASONING:"
        )

        print(
            result.reasoning
        )

        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())