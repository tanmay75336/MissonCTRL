from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.agent.evaluator import EvaluationResult, EvidenceEvaluator, NextActionDecision
from app.agent.events import ResearchEvent, ResearchEventType
from app.agent.research_executor import Evidence, ResearchExecutor
from app.agent.research_plan import (
    Claim, ClaimValidator, CoverageAnalyzer, CoverageMap, MissionDecomposer,
    NextResearchAction, ResearchPlan, ResearchPrioritizer, ResearchSearchRecord,
)
from app.agent.strategy_engine import ResearchStrategy, StrategyEngine
from app.agent.strategy_selector import StrategySelector
from app.agent.strategy_switcher import StrategySwitcher
from app.agent.synthesizer import FinalSynthesizer


class ResearchLoopResult(BaseModel):
    mission: str
    answer: str
    facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    sources: list[Any] = Field(default_factory=list)
    strategy_history: list[str] = Field(default_factory=list)
    searches_used: int = 0
    remaining_searches: int = 0
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    decisions: list[NextActionDecision] = Field(default_factory=list)
    events: list[ResearchEvent] = Field(default_factory=list)
    research_plan: ResearchPlan | None = None
    coverage: CoverageMap | None = None
    claims: list[Claim] = Field(default_factory=list)
    search_records: list[ResearchSearchRecord] = Field(default_factory=list)


class ResearchLoop:
    """Budget-bounded autonomous research directed by evidence sufficiency."""

    def __init__(
        self,
        executor: ResearchExecutor,
        strategy_engine: StrategyEngine,
        strategy_selector: StrategySelector,
        synthesizer: FinalSynthesizer,
        evaluator: EvidenceEvaluator | None = None,
        strategy_switcher: StrategySwitcher | None = None,
        event_callback: Callable[[ResearchEvent], None] | None = None,
    ):
        self.executor = executor
        self.strategy_engine = strategy_engine
        self.strategy_selector = strategy_selector
        self.synthesizer = synthesizer
        self.evaluator = evaluator or EvidenceEvaluator()
        self.strategy_switcher = strategy_switcher
        self.event_callback = event_callback
        self._events: list[ResearchEvent] = []

    async def run(self, mission: str, plan: str = "") -> ResearchLoopResult:
        if not mission.strip():
            raise ValueError("Mission cannot be empty.")

        self._events = []
        self._emit(
            "MISSION_STARTED",
            "Mission started.",
            {"mission": mission},
        )

        research_plan = MissionDecomposer().decompose(mission)
        coverage_analyzer = CoverageAnalyzer()
        prioritizer = ResearchPrioritizer()
        claim_validator = ClaimValidator()
        use_objective_plan = len(research_plan.research_objectives) > 1
        coverage = coverage_analyzer.map(research_plan.research_objectives)
        claims: list[Claim] = []
        search_records: list[ResearchSearchRecord] = []
        objective_evidence: dict[str, list[Evidence]] = {
            objective.id: [] for objective in research_plan.research_objectives
        }
        self._emit(
            "RESEARCH_PLAN_CREATED",
            "Created a research plan.",
            {"objective_count": len(research_plan.research_objectives), "time_sensitive": research_plan.time_sensitive},
        )

        print("\n========================================")
        print("MISSIONCTRL RESEARCH LOOP")
        print("========================================")
        strategy_set = await self.strategy_engine.generate(mission=mission, plan=plan)
        strategies = strategy_set.strategies
        if not strategies:
            raise RuntimeError("Strategy engine returned no strategies.")

        selection = await self.strategy_selector.select(mission=mission, strategies=strategies)
        current_strategy = self._find_strategy(strategies, selection.selected_strategy)
        if current_strategy is None:
            raise RuntimeError("Strategy selector returned a strategy that does not exist.")

        print(f"[STRATEGY] Selected: {current_strategy.name}")
        self._emit(
            "STRATEGY_SELECTED",
            "Selected research strategy.",
            {"strategy": current_strategy.name},
        )
        all_evidence: list[Evidence] = []
        evaluations: list[EvaluationResult] = []
        decisions: list[NextActionDecision] = []
        strategy_history = [current_strategy.name]
        used_strategy_names = {current_strategy.name}
        previous_questions: set[str] = set()
        previous_queries: set[str] = set()
        unresolved_questions: list[str] = []
        attempts_under_current_strategy = 0
        current_objective = prioritizer.select(research_plan.research_objectives) if use_objective_plan else None
        current_question = (
            self._next_objective_question(current_objective, previous_questions)
            if current_objective is not None
            else self._first_question(current_strategy)
        )

        while current_question:
            normalized_question = self._normalize_text(current_question)
            if normalized_question in previous_questions:
                self._append_unique(unresolved_questions, current_question)
                print("[RESEARCH] Duplicate research question prevented.")
                self._emit(
                    "MISSION_STOPPED",
                    "Stopped because a duplicate research question was prevented.",
                    {"question": current_question},
                )
                break
            if self.executor.budget.exhausted:
                print("[RESEARCH] Search budget exhausted.")
                self._emit(
                    "MISSION_STOPPED",
                    "Stopped because the search budget is exhausted.",
                    {},
                )
                break

            previous_questions.add(normalized_question)
            print(f"\n[QUESTION] {current_question}")
            if current_objective is not None:
                self._emit(
                    "OBJECTIVE_SELECTED",
                    "Selected the least-covered high-priority objective.",
                    {"objective_id": current_objective.id, "title": current_objective.title},
                )
            self._emit(
                "RESEARCH_QUESTION",
                "Research question selected.",
                {"question": current_question},
            )
            self._emit(
                "SEARCH_STARTED",
                "Executing research search.",
                {"question": current_question},
            )
            try:
                research_kwargs = {
                    "question": current_question,
                    "context": mission,
                    "previous_queries": previous_queries,
                }
                if current_objective is not None:
                    research_kwargs["objective_id"] = current_objective.id
                research_result = await self.executor.research(**research_kwargs)
            except RuntimeError as error:
                # The executor rejects exhausted budgets and duplicate queries
                # before a SerpApi call can be made.
                print(f"[RESEARCH] {error}")
                self._emit(
                    "MISSION_STOPPED",
                    "Research stopped before a search could be completed.",
                    {"reason": str(error)},
                )
                self._append_unique(unresolved_questions, current_question)
                break

            attempts_under_current_strategy += 1
            previous_queries.add(self._normalize_text(research_result.decision.query))
            all_evidence.extend(research_result.evidence)
            if current_objective is not None:
                objective_evidence[current_objective.id].extend(research_result.evidence)
                current_objective.status = "RESEARCHING"
                coverage_analyzer.update(
                    current_objective,
                    objective_evidence[current_objective.id],
                )
                coverage = coverage_analyzer.map(research_plan.research_objectives)
                claim = claim_validator.validate(
                    current_objective,
                    objective_evidence[current_objective.id],
                )
                claims = [item for item in claims if item.objective_id != claim.objective_id] + [claim]
                search_records.append(ResearchSearchRecord(
                    objective_id=current_objective.id,
                    question=current_question,
                    query=research_result.decision.query,
                    tool=research_result.decision.tool,
                    evidence_count=research_result.raw_evidence_count,
                    useful_evidence_count=research_result.filtered_evidence_count,
                ))
                self._emit(
                    "COVERAGE_UPDATED",
                    "Updated evidence coverage for the current objective.",
                    {"objective_id": current_objective.id, "coverage": current_objective.coverage_score, "overall_coverage": coverage.overall_coverage},
                )
                self._emit(
                    "CLAIM_VALIDATED",
                    "Validated the objective claim against collected evidence.",
                    {"objective_id": current_objective.id, "status": claim.status},
                )
                if claim.contradictions:
                    self._emit(
                        "CONTRADICTION_FOUND",
                        "Conflicting evidence remains visible for this objective.",
                        {"objective_id": current_objective.id},
                    )
            self._emit(
                "SEARCH_COMPLETED",
                "Research search completed.",
                {
                    "tool": research_result.decision.tool,
                    "query": research_result.decision.query,
                    "results": research_result.raw_evidence_count,
                },
            )
            self._emit(
                "EVIDENCE_COLLECTED",
                "Relevant evidence collected.",
                {"count": research_result.filtered_evidence_count},
            )
            rejected_count = (
                research_result.raw_evidence_count
                - research_result.filtered_evidence_count
            )
            if rejected_count:
                self._emit(
                    "EVIDENCE_REJECTED",
                    "Evidence was rejected by the critic.",
                    {"count": rejected_count},
                )

            # Local coverage answers whether this particular research step
            # was useful. The mission-level decision below still determines
            # whether the overall objective is sufficiently researched.
            question_evaluation = self.evaluator.evaluate(
                question=current_question,
                evidence=research_result.evidence,
            )
            evaluations.append(question_evaluation)

            mission_evaluation = self.evaluator.evaluate(
                question=mission,
                evidence=all_evidence,
            )
            evaluations.append(mission_evaluation)
            self._emit(
                "EVIDENCE_EVALUATED",
                "Evaluated local question and overall mission evidence.",
                {
                    "question_sufficient": question_evaluation.sufficient,
                    "mission_sufficient": mission_evaluation.sufficient,
                    "evidence_count": mission_evaluation.evidence_count,
                },
            )
            decision = self.evaluator.decide_next_action(
                question=mission,
                evidence=all_evidence,
                remaining_budget=self.executor.budget.remaining,
            )
            if use_objective_plan:
                decision, current_objective, current_question = self._objective_next_action(
                    decision=decision,
                    research_plan=research_plan,
                    coverage=coverage,
                    prioritizer=prioritizer,
                    previous_questions=previous_questions,
                    remaining_budget=self.executor.budget.remaining,
                    current_objective=current_objective,
                )
            decision = self._validate_next_action(
                decision=decision,
                mission=mission,
                current_question=current_question,
                previous_questions=previous_questions,
                mission_evaluation=mission_evaluation,
            )
            decisions.append(decision)
            self._emit(
                "DECISION",
                "Evidence sufficiency decision recorded.",
                {
                    "status": decision.status,
                    "next_action": decision.next_action,
                    "missing_information": decision.missing_information,
                    "next_question": decision.next_question,
                },
            )
            self._log_decision(decision)
            for item in decision.missing_information:
                self._append_unique(unresolved_questions, item)

            if decision.next_action == "STOP":
                self._emit(
                    "MISSION_STOPPED",
                    "Mission stopped with unresolved uncertainty.",
                    {"reason": decision.reason},
                )
                break

            if decision.next_action == "SYNTHESIZE":
                break

            if use_objective_plan and current_question:
                continue

            next_question = decision.next_question

            # A new search alone is not enough to justify a strategy switch.
            switch_result = await self._maybe_switch_strategy(
                current_strategy=current_strategy,
                strategies=strategies,
                used_strategy_names=used_strategy_names,
                evidence=all_evidence,
                unresolved_questions=unresolved_questions,
                attempts_under_current_strategy=attempts_under_current_strategy,
            )
            if switch_result is not None:
                switched, switch_reason = switch_result
                previous_strategy = current_strategy.name
                current_strategy = switched
                used_strategy_names.add(current_strategy.name)
                strategy_history.append(current_strategy.name)
                attempts_under_current_strategy = 0
                print(f"[STRATEGY SWITCH] Switched to: {current_strategy.name}")
                self._emit(
                    "STRATEGY_SWITCHED",
                    "Switched research strategy.",
                    {
                        "from": previous_strategy,
                        "to": current_strategy.name,
                        "reason": switch_reason,
                    },
                )
                current_question = self._next_strategy_question(
                    current_strategy,
                    previous_questions,
                )
                if not current_question:
                    current_question = next_question
            else:
                current_question = next_question

        if self.executor.budget.exhausted:
            self._append_unique(
                unresolved_questions,
                "Research budget exhausted before all uncertainty could be resolved.",
            )

        self._emit(
            "SYNTHESIS_STARTED",
            "Generating final evidence-grounded synthesis.",
            {},
        )
        synthesis_kwargs = {
            "mission": mission,
            "evidence": all_evidence,
            "unresolved_questions": unresolved_questions,
            "strategy_history": strategy_history,
        }
        # Legacy synthesizers used by integrations/tests remain compatible.
        if "research_plan" in inspect.signature(self.synthesizer.synthesize).parameters:
            synthesis_kwargs.update({
                "research_plan": research_plan,
                "claims": claims,
                "coverage": coverage,
            })
        final_synthesis = await self.synthesizer.synthesize(**synthesis_kwargs)
        result = ResearchLoopResult(
            mission=mission,
            answer=final_synthesis.answer,
            facts=final_synthesis.facts,
            inferences=final_synthesis.inferences,
            unknowns=final_synthesis.unknowns,
            unresolved_questions=list(dict.fromkeys(
                unresolved_questions + final_synthesis.unresolved_questions
            )),
            sources=final_synthesis.sources,
            strategy_history=strategy_history,
            searches_used=self.executor.budget.used,
            remaining_searches=self.executor.budget.remaining,
            evaluations=evaluations,
            decisions=decisions,
            events=list(self._events),
            research_plan=research_plan,
            coverage=coverage,
            claims=claims,
            search_records=search_records,
        )
        self._emit(
            "MISSION_COMPLETED",
            "Mission completed.",
            {
                "searches_used": result.searches_used,
                "remaining_searches": result.remaining_searches,
            },
        )
        result.events = list(self._events)
        return result

    async def _maybe_switch_strategy(
        self,
        current_strategy: ResearchStrategy,
        strategies: list[ResearchStrategy],
        used_strategy_names: set[str],
        evidence: list[Evidence],
        unresolved_questions: list[str],
        attempts_under_current_strategy: int,
    ) -> tuple[ResearchStrategy, str] | None:
        if self.strategy_switcher is None:
            return None
        available = [item.name for item in strategies if item.name not in used_strategy_names]
        if not available:
            return None
        relevant = [item for item in evidence if item.relevant]
        average_relevance = (
            sum(item.relevance_score for item in relevant) / len(relevant)
            if relevant else 0.0
        )
        decision = await self.strategy_switcher.decide(
            current_strategy=current_strategy.name,
            available_strategies=available,
            evidence_count=len(evidence),
            average_relevance=average_relevance,
            unresolved_questions=len(unresolved_questions),
            remaining_budget=self.executor.budget.remaining,
            attempts_under_current_strategy=attempts_under_current_strategy,
        )
        if not decision.should_switch:
            return None
        strategy = self._find_strategy(strategies, decision.target_strategy)
        if strategy is None:
            return None
        return strategy, decision.reason

    @staticmethod
    def _first_question(strategy: ResearchStrategy) -> str:
        return next((item for item in strategy.research_questions if item.strip()), "")

    def _next_strategy_question(
        self,
        strategy: ResearchStrategy,
        previous_questions: set[str],
    ) -> str:
        """Return the first unused meaningful question in a new strategy."""

        for question in strategy.research_questions:
            if (
                question.strip()
                and self._normalize_text(question)
                not in previous_questions
            ):
                return question
        return ""

    def _next_objective_question(
        self,
        objective,
        previous_questions: set[str],
    ) -> str:
        for question in objective.research_questions:
            if self._normalize_text(question) not in previous_questions:
                return question
        return ""

    def _objective_next_action(
        self,
        decision: NextActionDecision,
        research_plan: ResearchPlan,
        coverage: CoverageMap,
        prioritizer: ResearchPrioritizer,
        previous_questions: set[str],
        remaining_budget: int,
        current_objective,
    ) -> tuple[NextActionDecision, Any, str]:
        if remaining_budget <= 0:
            unresolved = [item.title for item in research_plan.research_objectives if item.status != "SUFFICIENT"]
            return NextActionDecision(
                status="INSUFFICIENT",
                reason="The search budget is exhausted before all research objectives were covered.",
                missing_information=unresolved,
                next_action="STOP",
                confidence=1.0,
            ), current_objective, ""

        candidates = sorted(
            [item for item in research_plan.research_objectives if item.status != "SUFFICIENT"],
            key=lambda item: (item.priority * (1 - item.coverage_score), item.priority),
            reverse=True,
        )
        for objective in candidates:
            question = self._next_objective_question(objective, previous_questions)
            if question:
                action = NextResearchAction(
                    objective_id=objective.id,
                    reason="This objective has the highest priority coverage gap.",
                    missing=objective.unresolved_gaps,
                    next_question=question,
                )
                self._emit(
                    "GAP_IDENTIFIED",
                    "Identified the next evidence gap.",
                    {"objective_id": action.objective_id, "missing": action.missing},
                )
                self._emit(
                    "NEXT_ACTION_SELECTED",
                    "Selected the next highest-value research action.",
                    {"objective_id": action.objective_id, "question": action.next_question},
                )
                return NextActionDecision(
                    status="INSUFFICIENT",
                    reason=action.reason,
                    missing_information=action.missing,
                    next_action="SEARCH",
                    next_question=action.next_question,
                    confidence=0.8,
                ), objective, question

        if all(item.status == "SUFFICIENT" for item in research_plan.research_objectives):
            return NextActionDecision(
                status="SUFFICIENT",
                reason="All high-priority research objectives have sufficient evidence coverage.",
                next_action="SYNTHESIZE",
                confidence=0.9,
            ), current_objective, ""

        unresolved = [item.title for item in research_plan.research_objectives if item.status != "SUFFICIENT"]
        return NextActionDecision(
            status="INSUFFICIENT",
            reason="No meaningful new question remains for the unresolved research objectives.",
            missing_information=unresolved,
            next_action="STOP",
            confidence=1.0,
        ), current_objective, ""

    def _validate_next_action(
        self,
        decision: NextActionDecision,
        mission: str,
        current_question: str,
        previous_questions: set[str],
        mission_evaluation: EvaluationResult,
    ) -> NextActionDecision:
        """Ensure a follow-up is a new, concrete research action.

        A duplicate or broad mission wrapper is not an error condition. We
        try the next deterministic evidence gap; if none is available, the
        safe autonomous action is to stop and synthesize with uncertainty.
        """

        if decision.next_action != "SEARCH":
            return decision

        if self._is_valid_next_question(
            candidate=decision.next_question,
            mission=mission,
            current_question=current_question,
            previous_questions=previous_questions,
        ):
            return decision

        excluded = set(previous_questions)
        excluded.add(self._normalize_text(current_question))
        excluded.add(self._normalize_text(mission))
        excluded.add(self._normalize_text(decision.next_question))
        fallback = self.evaluator.create_targeted_follow_up(
            mission=mission,
            evaluation=mission_evaluation,
            excluded_questions=excluded,
        )
        if self._is_valid_next_question(
            candidate=fallback,
            mission=mission,
            current_question=current_question,
            previous_questions=previous_questions,
        ):
            return decision.model_copy(update={"next_question": fallback})

        missing = list(decision.missing_information)
        self._append_unique(
            missing,
            "No new specific research question could be generated.",
        )
        return NextActionDecision(
            status="INSUFFICIENT",
            reason=(
                "Evidence remains insufficient, but no materially new "
                "research question is available."
            ),
            missing_information=missing,
            next_action="STOP",
            confidence=1.0,
        )

    def _is_valid_next_question(
        self,
        candidate: str,
        mission: str,
        current_question: str,
        previous_questions: set[str],
    ) -> bool:
        normalized = self._normalize_text(candidate)
        if not normalized or normalized in previous_questions:
            return False

        mission_normalized = self._normalize_text(mission)
        current_normalized = self._normalize_text(current_question)
        if normalized in {mission_normalized, current_normalized}:
            return False

        generic_wrappers = (
            "find independent sources that validate the findings about",
            "find additional independent evidence about",
            "find evidence that directly answers",
        )
        if any(wrapper in normalized for wrapper in generic_wrappers):
            return False

        return not (
            self._substantially_equivalent(normalized, mission_normalized)
            or self._substantially_equivalent(normalized, current_normalized)
        )

    @staticmethod
    def _substantially_equivalent(left: str, right: str) -> bool:
        left_words = set(left.split())
        right_words = set(right.split())
        if not left_words or not right_words:
            return False
        return len(left_words & right_words) / min(len(left_words), len(right_words)) >= 0.8

    @staticmethod
    def _find_strategy(strategies: list[ResearchStrategy], name: str) -> ResearchStrategy | None:
        return next((item for item in strategies if item.name == name), None)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(
            "".join(character if character.isalnum() else " " for character in text.lower()).split()
        )

    @staticmethod
    def _append_unique(items: list[str], value: str) -> None:
        if value.strip() and value not in items:
            items.append(value)

    @staticmethod
    def _log_decision(decision: NextActionDecision) -> None:
        print(f"[DECISION] {decision.status}")
        print(f"[REASON] {decision.reason}")
        if decision.missing_information:
            print(f"[MISSING] {', '.join(decision.missing_information)}")
        print(f"[NEXT ACTION] {decision.next_action}")
        if decision.next_question:
            print(f"[NEXT QUESTION] {decision.next_question}")

    def _emit(
        self,
        event_type: ResearchEventType,
        message: str,
        data: dict[str, Any],
    ) -> None:
        event = ResearchEvent(type=event_type, message=message, data=data)
        self._events.append(event)
        if self.event_callback is not None:
            try:
                self.event_callback(event)
            except Exception:
                # Observability must not interrupt autonomous research.
                pass
