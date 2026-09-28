import asyncio

from app.agent.evidence_critic import EvidenceCritic
from app.agent.research_executor import Evidence
from app.llm.factory import get_llm_provider


async def main():

    llm = get_llm_provider("groq")

    critic = EvidenceCritic(
        llm=llm
    )

    question = (
        "What unmet needs or pain points do "
        "students in Mumbai currently experience?"
    )

    evidence_items = [

        Evidence(
            title=(
                "After suicide case, IIT Bombay students "
                "seek welfare reforms"
            ),
            url=(
                "https://example.com/iit-bombay"
            ),
            snippet=(
                "Students have raised concerns about "
                "student welfare and mental health support."
            ),
            source="Hindustan Times",
            tool="web_search",
            query=question,
        ),

        Evidence(
            title=(
                "Caliber Mining IPO 2026: Complete Review"
            ),
            url=(
                "https://example.com/caliber-mining"
            ),
            snippet=(
                "IPO price band, allotment date and "
                "subscription details."
            ),
            source="India Infoline",
            tool="news_search",
            query=question,
        ),

        Evidence(
            title=(
                "Top 21 Franchise Business in India"
            ),
            url=(
                "https://example.com/franchises"
            ),
            snippet=(
                "List of franchise business opportunities "
                "available in India."
            ),
            source="SugarMint",
            tool="web_search",
            query=question,
        ),
    ]

    print("\n" + "=" * 60)
    print("MISSIONCTRL — EVIDENCE CRITIC TEST")
    print("=" * 60)

    for index, evidence in enumerate(
        evidence_items,
        start=1,
    ):

        print(
            f"\nEVALUATING EVIDENCE #{index}"
        )

        print(
            f"TITLE: {evidence.title}"
        )

        result = await critic.assess(
            question=question,
            evidence=evidence,
        )

        print(
            f"RELEVANT: {result.relevant}"
        )

        print(
            f"SCORE: {result.relevance_score}"
        )

        print(
            f"CLAIM: {result.claim}"
        )

        print(
            f"SOURCE QUALITY: "
            f"{result.source_quality}"
        )

        print(
            f"REASONING: {result.reasoning}"
        )

        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())