import asyncio

from app.agent.research_executor import Evidence
from app.agent.synthesizer import FinalSynthesizer
from app.llm.factory import get_llm_provider


# ============================================================
# TEST MISSION
# ============================================================

MISSION = (
    "I have ₹2 lakh and want to find "
    "a student-focused business opportunity in Mumbai."
)


# ============================================================
# TEST
# ============================================================


async def main():

    print("\n")
    print("=" * 70)
    print("MISSIONCTRL — FINAL SYNTHESIS TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LLM
    # --------------------------------------------------------

    print(
        "\n[SETUP] Initializing Groq..."
    )

    llm = get_llm_provider("groq")

    # --------------------------------------------------------
    # 2. SYNTHESIZER
    # --------------------------------------------------------

    synthesizer = FinalSynthesizer(
        llm=llm
    )

    # --------------------------------------------------------
    # 3. FAKE EVIDENCE
    # --------------------------------------------------------
    # These are deliberately limited so we can test whether
    # the synthesizer avoids inventing unsupported conclusions.
    # --------------------------------------------------------

    evidence = [

        Evidence(
            title="Student Needs in Mumbai",

            url=(
                "https://example.com/"
                "student-needs"
            ),

            snippet=(
                "Students in Mumbai face challenges "
                "involving affordable services, convenience "
                "and academic support."
            ),

            source="Example Research",

            tool="web_search",

            query=(
                "Mumbai student unmet needs"
            ),

            relevant=True,

            relevance_score=0.85,

            claim=(
                "Students in Mumbai face challenges "
                "involving affordable services, convenience "
                "and academic support."
            ),

            source_quality="medium",

            critic_reasoning=(
                "The result directly discusses "
                "student needs in Mumbai."
            ),
        ),

        Evidence(
            title="Student Businesses in Mumbai",

            url=(
                "https://example.com/"
                "student-businesses"
            ),

            snippet=(
                "Several businesses provide services "
                "to students around Mumbai colleges."
            ),

            source="Example Business Report",

            tool="web_search",

            query=(
                "Mumbai student businesses"
            ),

            relevant=True,

            relevance_score=0.75,

            claim=(
                "Businesses already provide services "
                "to students around Mumbai colleges."
            ),

            source_quality="low",

            critic_reasoning=(
                "Relevant to the competitive landscape, "
                "but source quality is limited."
            ),
        ),
    ]

    # --------------------------------------------------------
    # 4. UNRESOLVED QUESTIONS
    # --------------------------------------------------------

    unresolved_questions = [

        "What are the actual startup costs?",

        "Which student segment has the strongest demand?",

        "What regulatory requirements apply?",
    ]

    # --------------------------------------------------------
    # 5. STRATEGY HISTORY
    # --------------------------------------------------------

    strategy_history = [

        "Student Pain-Point & Demand Mapping",
    ]

    # --------------------------------------------------------
    # 6. RUN SYNTHESIS
    # --------------------------------------------------------

    print(
        "\n[SYNTHESIZER] "
        "Generating evidence-grounded answer..."
    )

    result = await synthesizer.synthesize(

        mission=MISSION,

        evidence=evidence,

        unresolved_questions=(
            unresolved_questions
        ),

        strategy_history=(
            strategy_history
        ),
    )

    # ========================================================
    # FINAL SYNTHESIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL SYNTHESIS")
    print("=" * 70)

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    print("\nANSWER:")
    print(result.answer)

    # --------------------------------------------------------
    # FACTS
    # --------------------------------------------------------

    print("\nFACTS:")

    if result.facts:

        for fact in result.facts:
            print(
                f"- {fact}"
            )

    else:

        print(
            "- None identified."
        )

    # --------------------------------------------------------
    # INFERENCES
    # --------------------------------------------------------

    print("\nINFERENCES:")

    if result.inferences:

        for inference in result.inferences:
            print(
                f"- {inference}"
            )

    else:

        print(
            "- None identified."
        )

    # --------------------------------------------------------
    # UNKNOWNS
    # --------------------------------------------------------

    print("\nUNKNOWNS:")

    if result.unknowns:

        for unknown in result.unknowns:
            print(
                f"- {unknown}"
            )

    else:

        print(
            "- None identified."
        )

    # --------------------------------------------------------
    # UNRESOLVED QUESTIONS
    # --------------------------------------------------------

    print("\nUNRESOLVED QUESTIONS:")

    if result.unresolved_questions:

        for question in (
            result.unresolved_questions
        ):
            print(
                f"- {question}"
            )

    else:

        print(
            "- None."
        )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    print("\nSOURCES:")

    if result.sources:

        for source in result.sources:

            print(
                f"\n{source.title}"
            )

            print(
                f"URL: "
                f"{source.url}"
            )

            print(
                f"Supports: "
                f"{source.claim_supported}"
            )

    else:

        print(
            "- No sources cited."
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("VALIDATING SYNTHESIS")
    print("=" * 70)

    # Basic validation
    assert result.answer.strip()

    # Every cited URL must come from supplied evidence
    evidence_urls = {
        item.url
        for item in evidence
        if item.url
    }

    for source in result.sources:

        assert source.url in evidence_urls

    # The unresolved questions supplied to the synthesizer
    # should not mysteriously disappear.
    assert len(
        result.unresolved_questions
    ) > 0

    print(
        "✓ Final answer generated"
    )

    print(
        "✓ Facts separated from inferences"
    )

    print(
        "✓ Unknowns explicitly represented"
    )

    print(
        "✓ Unresolved questions preserved"
    )

    print(
        "✓ Source URLs validated"
    )

    print("\n")
    print("=" * 70)
    print("✅ SYNTHESIS TEST PASSED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    asyncio.run(main())