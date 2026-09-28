"""One-run, opt-in autonomous mission integration test.

Run only with:
    $env:RUN_REAL_AUTONOMOUS_MISSION = "1"
    python -m pytest -v -s test_real_autonomous_mission.py
"""

import os

import pytest

from app.agent.budget import SearchBudget
from app.agent.evaluator import EvidenceEvaluator
from app.agent.research_executor import ResearchExecutor
from app.agent.research_loop import ResearchLoop
from app.agent.strategy_engine import StrategyEngine
from app.agent.strategy_selector import StrategySelector
from app.agent.strategy_switcher import StrategySwitcher
from app.agent.synthesizer import FinalSynthesizer
from app.config import settings
from app.llm.groq import GroqProvider
from app.tools.serpapi import SerpApiTools


pytestmark = pytest.mark.integration

MISSION = (
    "Research the SerpApi India Hackathon 2026 and identify official "
    "submission requirements, official judging criteria, and important "
    "rules or constraints participants must follow."
)


class RecordingResearchExecutor(ResearchExecutor):
    """The production executor with an in-memory record of completed searches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completed_searches = []

    async def research(self, *args, **kwargs):
        result = await super().research(*args, **kwargs)
        self.completed_searches.append(result)
        return result


def _print_search(number, result) -> None:
    print(f"\n[SEARCH {number}]")
    print(f"Tool: {result.decision.tool}")
    print(f"Query: {result.decision.query}")
    print(f"Results: {result.raw_evidence_count}")
    print(f"Relevant evidence: {result.filtered_evidence_count}")


def _print_decision(number, decision) -> None:
    print(f"\n[DECISION {number}]")
    print(f"Status: {decision.status}")
    print(f"Reason: {decision.reason}")
    print(f"Missing information: {', '.join(decision.missing_information) or 'None'}")
    print(f"Next action: {decision.next_action}")
    if decision.next_question:
        print(f"Next question: {decision.next_question}")


@pytest.mark.anyio
@pytest.mark.skipif(
    os.getenv("RUN_REAL_AUTONOMOUS_MISSION") != "1",
    reason="Set RUN_REAL_AUTONOMOUS_MISSION=1 to authorize up to two live SerpApi searches.",
)
async def test_real_autonomous_mission() -> None:
    """Run one evidence-guided mission with an absolute two-search cap."""

    # Presence checks deliberately never reveal secret values.
    assert settings.serpapi_api_key
    assert settings.groq_api_key
    assert settings.default_llm.lower() == "groq"

    llm = GroqProvider()
    budget = SearchBudget(maximum=2)
    executor = RecordingResearchExecutor(
        serpapi_tools=SerpApiTools(),
        llm=llm,
        budget=budget,
    )
    loop = ResearchLoop(
        executor=executor,
        strategy_engine=StrategyEngine(llm),
        strategy_selector=StrategySelector(llm),
        synthesizer=FinalSynthesizer(llm),
        evaluator=EvidenceEvaluator(),
        strategy_switcher=StrategySwitcher(llm),
    )

    result = await loop.run(mission=MISSION)

    print("\n[MISSION]")
    print(MISSION)
    print("\n[STRATEGY]")
    print(" -> ".join(result.strategy_history))
    for number, search in enumerate(executor.completed_searches, start=1):
        _print_search(number, search)
    for number, decision in enumerate(result.decisions, start=1):
        _print_decision(number, decision)

    queries = [search.decision.query for search in executor.completed_searches]
    normalized_queries = [executor._normalize_search_text(query) for query in queries]
    retrieved_urls = {
        item.url
        for search in executor.completed_searches
        for item in search.evidence
        if item.url
    }

    assert len(executor.completed_searches) <= 2
    assert budget.used <= 2
    assert budget.remaining >= 0
    assert len(normalized_queries) == len(set(normalized_queries))
    assert result.answer.strip()
    assert all(source.url in retrieved_urls for source in result.sources)

    if result.decisions:
        final_decision = result.decisions[-1]
        if final_decision.next_action == "SYNTHESIZE":
            assert final_decision.status == "SUFFICIENT"
        if final_decision.next_action == "STOP":
            assert result.unresolved_questions

    print("\n[FINAL ANSWER]")
    print(result.answer)
    print("\n[UNRESOLVED QUESTIONS]")
    for question in result.unresolved_questions:
        print(f"- {question}")
    print("\n[SOURCES]")
    for source in result.sources:
        print(f"- {source.title}: {source.url}")
    print("\n[SAFETY]")
    print(f"Searches used: {budget.used}")
    print(f"Remaining budget: {budget.remaining}")
    print("API key exposed: NO")
