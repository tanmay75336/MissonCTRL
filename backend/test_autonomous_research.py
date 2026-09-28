"""Fake-only coverage for autonomous evidence sufficiency decisions."""

import pytest

from app.agent.budget import SearchBudget
from app.agent.evaluator import EvidenceEvaluator, NextActionDecision
from app.agent.research_executor import Evidence, ResearchResult
from app.agent.research_loop import ResearchLoop
from app.agent.strategy_engine import ResearchStrategy, StrategySet
from app.agent.strategy_selector import StrategySelection
from app.agent.strategy_switcher import StrategySwitchDecision
from app.agent.synthesizer import FinalSynthesis
from app.agent.tool_router import ToolDecision


def evidence(
    title: str,
    source: str,
    snippet: str,
    *,
    quality: str = "medium",
    metadata: dict | None = None,
) -> Evidence:
    return Evidence(
        title=title,
        url=f"https://{source.lower().replace(' ', '-')}.example/article",
        snippet=snippet,
        source=source,
        tool="web_search",
        query="hackathon registration deadline",
        source_quality=quality,
        relevance_score=0.9,
        metadata=metadata or {},
    )


def test_strong_relevant_evidence_synthesizes() -> None:
    item = evidence(
        "Official registration deadline",
        "Official Hackathon",
        "Registration closes on 30 June.",
        quality="high",
    )

    decision = EvidenceEvaluator().decide_next_action(
        question="When does registration close?",
        evidence=[item],
        remaining_budget=2,
    )

    assert decision.status == "SUFFICIENT"
    assert decision.next_action == "SYNTHESIZE"


def test_weak_evidence_requests_purposeful_search() -> None:
    decision = EvidenceEvaluator().decide_next_action(
        question="What are the official judging criteria?",
        evidence=[evidence("Unrelated event", "Blog", "A music event happened.")],
        remaining_budget=2,
    )

    assert decision.status == "INSUFFICIENT"
    assert decision.next_action == "SEARCH"
    assert decision.next_question
    assert decision.missing_information


def test_follow_up_targets_a_specific_mission_gap() -> None:
    mission = (
        "Compare NVIDIA AI accelerators with AMD MI series market share, "
        "product offerings, and semiconductor supply chain constraints."
    )
    decision = EvidenceEvaluator().decide_next_action(
        question=mission,
        evidence=[],
        remaining_budget=2,
    )

    assert decision.next_action == "SEARCH"
    assert "market share" in decision.next_question.lower()
    assert "independent sources that validate" not in decision.next_question.lower()
    assert any("market share" in gap.lower() for gap in decision.missing_information)


def test_budget_exhaustion_stops_without_search() -> None:
    decision = EvidenceEvaluator().decide_next_action(
        question="What are the official judging criteria?",
        evidence=[],
        remaining_budget=0,
    )

    assert decision.status == "INSUFFICIENT"
    assert decision.next_action == "STOP"
    assert not decision.next_question


def test_contradictory_evidence_is_not_sufficient() -> None:
    items = [
        evidence(
            "Official deadline", "Official Site", "The deadline is 1 June.",
            quality="high", metadata={"contradiction": "Deadline conflicts with another source."},
        ),
        evidence("Deadline update", "News", "The deadline is 15 June."),
    ]

    decision = EvidenceEvaluator().decide_next_action(
        question="When does registration close?",
        evidence=items,
        remaining_budget=1,
    )

    assert decision.status == "INSUFFICIENT"
    assert decision.next_action == "SEARCH"
    assert any("contradiction" in item.lower() for item in decision.missing_information)


class _StrategyEngine:
    async def generate(self, mission: str, plan: str) -> StrategySet:
        return StrategySet(strategies=[
            ResearchStrategy(
                name="Official sources",
                objective="Find official information.",
                approach="Search official sources first.",
                research_questions=["What are the official judging criteria?"],
            ),
        ])


class _StrategySelector:
    async def select(self, mission: str, strategies: list[ResearchStrategy]) -> StrategySelection:
        return StrategySelection(
            selected_strategy="Official sources",
            evaluations=[],
            reason="It is the most direct strategy.",
        )


class _Synthesizer:
    async def synthesize(self, mission, evidence, unresolved_questions, strategy_history):
        return FinalSynthesis(
            answer="Answer based only on supplied evidence.",
            facts=[item.snippet for item in evidence],
            unknowns=unresolved_questions,
        )


class _ScriptedExecutor:
    def __init__(self, batches: list[list[Evidence]], maximum: int = 2):
        self.batches = batches
        self.budget = SearchBudget(maximum=maximum)
        self.calls: list[str] = []

    async def research(self, question: str, context: str, previous_queries: set[str], objective_id: str = ""):
        query = f"query-{len(self.calls) + 1}"
        assert query not in previous_queries
        self.calls.append(question)
        self.budget.consume()
        items = self.batches[len(self.calls) - 1]
        return ResearchResult(
            question=question,
            decision=ToolDecision(tool="web_search", query=query, reason="fake"),
            evidence=items,
            raw_evidence_count=len(items),
            filtered_evidence_count=len(items),
            search_number=self.budget.used,
            remaining_searches=self.budget.remaining,
            objective_id=objective_id,
        )


class _RepeatedQuestionEvaluator(EvidenceEvaluator):
    def decide_next_action(self, question, evidence, remaining_budget, contradictions=None):
        return NextActionDecision(
            status="INSUFFICIENT",
            reason="More evidence is needed.",
            next_action="SEARCH",
            next_question="What are the official judging criteria?",
            confidence=0.8,
        )


class _GenericFollowUpEvaluator(EvidenceEvaluator):
    def __init__(self):
        self.calls = 0

    def decide_next_action(self, question, evidence, remaining_budget, contradictions=None):
        self.calls += 1
        if self.calls > 1:
            return NextActionDecision(
                status="SUFFICIENT",
                reason="The follow-up provided enough coverage.",
                next_action="SYNTHESIZE",
                confidence=0.9,
            )
        return NextActionDecision(
            status="INSUFFICIENT",
            reason="Generic evidence is insufficient.",
            next_action="SEARCH",
            next_question=f"Find independent sources that validate the findings about: {question}",
            confidence=0.8,
        )


class _NoNewQuestionEvaluator(_GenericFollowUpEvaluator):
    def create_targeted_follow_up(self, mission, evaluation, excluded_questions=None):
        return ""


class _SwitchThenSynthesizeEvaluator(EvidenceEvaluator):
    def __init__(self, fallback_question: str):
        self.fallback_question = fallback_question
        self.decisions_made = 0

    def decide_next_action(self, question, evidence, remaining_budget, contradictions=None):
        self.decisions_made += 1
        if self.decisions_made == 1:
            return NextActionDecision(
                status="INSUFFICIENT",
                reason="An official source is still needed.",
                next_action="SEARCH",
                next_question=self.fallback_question,
                confidence=0.8,
            )
        return NextActionDecision(
            status="SUFFICIENT",
            reason="The mission is now covered.",
            next_action="SYNTHESIZE",
            confidence=0.9,
        )


class _TwoStrategyEngine:
    def __init__(self, strategy_b_question: str):
        self.strategy_b_question = strategy_b_question

    async def generate(self, mission: str, plan: str) -> StrategySet:
        return StrategySet(strategies=[
            ResearchStrategy(
                name="General sources",
                objective="Find general information.",
                approach="Search broadly.",
                research_questions=["Find general information."],
            ),
            ResearchStrategy(
                name="Official sources",
                objective="Find primary sources.",
                approach="Search official sources.",
                research_questions=[self.strategy_b_question],
            ),
        ])


class _GeneralSelector:
    async def select(self, mission: str, strategies: list[ResearchStrategy]) -> StrategySelection:
        return StrategySelection(
            selected_strategy="General sources",
            evaluations=[],
            reason="Start broad, then validate with primary sources.",
        )


class _AlwaysSwitch:
    async def decide(self, **kwargs) -> StrategySwitchDecision:
        return StrategySwitchDecision(
            should_switch=True,
            target_strategy="Official sources",
            reason="Official sources are more likely to resolve the gap.",
            confidence=0.9,
        )


@pytest.mark.anyio
async def test_repeated_research_question_is_not_executed_twice() -> None:
    executor = _ScriptedExecutor([
        [evidence("Criteria", "Official", "Criteria exist.")],
        [evidence("Primary criteria", "Official 2", "Criteria are documented.")],
    ])
    loop = ResearchLoop(
        executor=executor,
        strategy_engine=_StrategyEngine(),
        strategy_selector=_StrategySelector(),
        synthesizer=_Synthesizer(),
        evaluator=_RepeatedQuestionEvaluator(),
    )

    result = await loop.run("Find official judging criteria.")

    assert len(executor.calls) == 2
    assert len({ResearchLoop._normalize_text(item) for item in executor.calls}) == 2
    assert result.searches_used == 2


@pytest.mark.anyio
async def test_generic_follow_up_is_replaced_with_specific_question() -> None:
    executor = _ScriptedExecutor([
        [evidence("Market overview", "Source A", "Market share evidence.")],
        [evidence("Product comparison", "Source B", "Product offering evidence.")],
    ])
    mission = "Compare market share and product offerings for Alpha and Beta."
    loop = ResearchLoop(
        executor=executor,
        strategy_engine=_StrategyEngine(),
        strategy_selector=_StrategySelector(),
        synthesizer=_Synthesizer(),
        evaluator=_GenericFollowUpEvaluator(),
    )

    result = await loop.run(mission)

    assert len(executor.calls) == 2
    assert "independent sources that validate" not in executor.calls[1].lower()
    assert "market share" in executor.calls[1].lower()
    assert result.decisions[0].next_question == executor.calls[1]


@pytest.mark.anyio
async def test_no_unique_follow_up_stops_and_synthesizes() -> None:
    executor = _ScriptedExecutor([[evidence("Overview", "Source", "Weak evidence.")]])
    loop = ResearchLoop(
        executor=executor,
        strategy_engine=_StrategyEngine(),
        strategy_selector=_StrategySelector(),
        synthesizer=_Synthesizer(),
        evaluator=_NoNewQuestionEvaluator(),
    )

    result = await loop.run("Find official information.")

    assert len(executor.calls) == 1
    assert result.decisions[-1].next_action == "STOP"
    assert result.answer
    assert any("No new specific" in item for item in result.unresolved_questions)


@pytest.mark.anyio
async def test_strategy_switch_uses_new_strategy_question() -> None:
    official_question = "Find official primary sources."
    executor = _ScriptedExecutor([
        [evidence("General overview", "Blog", "General information.")],
        [evidence("Official criteria", "Official", "Official criteria are published.")],
    ])
    loop = ResearchLoop(
        executor=executor,
        strategy_engine=_TwoStrategyEngine(official_question),
        strategy_selector=_GeneralSelector(),
        synthesizer=_Synthesizer(),
        evaluator=_SwitchThenSynthesizeEvaluator("Evaluator fallback question."),
        strategy_switcher=_AlwaysSwitch(),
    )

    result = await loop.run("Find official information.")

    assert executor.calls == ["Find general information.", official_question]
    assert result.strategy_history == ["General sources", "Official sources"]
    assert result.evaluations[0].question == "Find general information."
    assert result.evaluations[1].question == "Find official information."


@pytest.mark.anyio
async def test_strategy_switch_falls_back_when_new_strategy_has_no_unused_question() -> None:
    fallback_question = "Locate primary documentation for the requirements."
    executor = _ScriptedExecutor([
        [evidence("General overview", "Blog", "General information.")],
        [evidence("Official source", "Official", "Official information.")],
    ])
    loop = ResearchLoop(
        executor=executor,
        # Strategy B repeats Strategy A's question, so it has no unused question.
        strategy_engine=_TwoStrategyEngine("Find general information."),
        strategy_selector=_GeneralSelector(),
        synthesizer=_Synthesizer(),
        evaluator=_SwitchThenSynthesizeEvaluator(fallback_question),
        strategy_switcher=_AlwaysSwitch(),
    )

    await loop.run("Find official information.")

    assert executor.calls == ["Find general information.", fallback_question]


@pytest.mark.anyio
async def test_autonomous_loop_searches_again_then_synthesizes() -> None:
    first_batch = [evidence("Criteria overview", "Official", "Judging criteria are published.")]
    second_batch = [
        evidence("Judging criteria details", "News A", "Judging criteria score originality."),
        evidence("Judging criteria rules", "News B", "Judging criteria require a demo."),
    ]
    executor = _ScriptedExecutor([first_batch, second_batch])
    received_events = []
    loop = ResearchLoop(
        executor=executor,
        strategy_engine=_StrategyEngine(),
        strategy_selector=_StrategySelector(),
        synthesizer=_Synthesizer(),
        event_callback=received_events.append,
    )

    result = await loop.run("What are the official judging criteria?")

    assert len(executor.calls) == 2
    assert result.searches_used == 2
    assert result.decisions[0].next_action == "SEARCH"
    assert result.decisions[-1].next_action == "SYNTHESIZE"
    event_types = {event.type for event in result.events}
    assert {
        "MISSION_STARTED",
        "STRATEGY_SELECTED",
        "SEARCH_STARTED",
        "SEARCH_COMPLETED",
        "EVIDENCE_EVALUATED",
        "DECISION",
        "SYNTHESIS_STARTED",
        "MISSION_COMPLETED",
    } <= event_types
    assert received_events == result.events
