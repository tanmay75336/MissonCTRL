from app.agent.tool_router import ToolRouter
from app.agent.budget import SearchBudget


def main():

    router = ToolRouter()
    budget = SearchBudget(maximum=8)

    questions = [
        "What unmet needs do students in Mumbai have?",
        "What are the latest student business trends in Mumbai?",
        "Which student-focused businesses are near Mumbai colleges?",
        "How has demand for student services changed over time?",
    ]

    print("\n" + "=" * 60)
    print("MISSIONCTRL — TOOL ROUTER TEST")
    print("=" * 60)

    for question in questions:

        decision = router.route(question)

        print("\nQUESTION:")
        print(question)

        print("\nTOOL:")
        print(decision.tool)

        print("\nQUERY:")
        print(decision.query)

        print("\nREASON:")
        print(decision.reason)

        print("-" * 60)

    print("\nSEARCH BUDGET")
    print(budget.status())


if __name__ == "__main__":
    main()