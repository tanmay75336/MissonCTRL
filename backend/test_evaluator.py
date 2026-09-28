from app.agent.evaluator import EvidenceEvaluator
from app.agent.research_executor import Evidence


def main():

    evaluator = EvidenceEvaluator()

    evidence = [
        Evidence(
            title="Mumbai Students Need Affordable Services",
            url="https://example.com/article1",
            snippet=(
                "Students in Mumbai are looking for "
                "affordable services and convenient options."
            ),
            source="Example News",
            tool="web_search",
            query="student needs Mumbai",
        ),
        Evidence(
            title="Student Market Research Mumbai",
            url="https://example.org/research",
            snippet=(
                "The Mumbai student market has demand "
                "for affordable products and services."
            ),
            source="Example Research",
            tool="web_search",
            query="student needs Mumbai",
        ),
        Evidence(
            title="Mumbai College Student Survey",
            url="https://example.net/survey",
            snippet=(
                "College students reported several "
                "unmet needs related to convenience."
            ),
            source="Example Survey",
            tool="web_search",
            query="student needs Mumbai",
        ),
    ]

    question = (
        "What unmet needs do students in Mumbai have?"
    )

    result = evaluator.evaluate(
        question=question,
        evidence=evidence,
    )

    print("\n" + "=" * 60)
    print("MISSIONCTRL — EVIDENCE EVALUATOR TEST")
    print("=" * 60)

    print("\nQUESTION:")
    print(result.question)

    print("\nEVIDENCE COUNT:")
    print(result.evidence_count)

    print("\nUNIQUE SOURCES:")
    print(result.unique_sources)

    print("\nRELEVANT EVIDENCE:")
    print(result.has_relevant_evidence)

    print("\nSOURCE DIVERSITY:")
    print(result.has_source_diversity)

    print("\nSUFFICIENT:")
    print(result.sufficient)

    print("\nMISSING INFORMATION:")

    for item in result.missing_information:
        print(f"- {item}")

    print("\nFOLLOW-UP QUESTIONS:")

    for item in result.follow_up_questions:
        print(f"- {item}")

    print("\nREASON:")
    print(result.reason)


if __name__ == "__main__":
    main()