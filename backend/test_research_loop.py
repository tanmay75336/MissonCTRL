import json

import pytest

from app.agent.budget import SearchBudget
from app.agent.evaluator import EvidenceEvaluator
from app.agent.research_executor import ResearchExecutor
from app.agent.research_loop import ResearchLoop
from app.agent.strategy_engine import StrategyEngine
from app.agent.strategy_selector import StrategySelector
from app.agent.strategy_switcher import StrategySwitcher
from app.agent.synthesizer import FinalSynthesizer


# ============================================================
# MISSION
# ============================================================

MISSION = (
    "I have ₹2 lakh and want to find a student-focused "
    "business opportunity in Mumbai."
)


# ============================================================
# FAKE LLM
# ============================================================

class FakeLLM:
    """
    Completely deterministic fake LLM.

    No Gemini.
    No Groq.
    No NVIDIA.
    No internet.

    This exists only to test the complete agent pipeline.
    """

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        system = system_prompt.lower()
        user = user_prompt.lower()

        # ====================================================
        # 1. STRATEGY ENGINE
        # ====================================================

        # Actual StrategyEngine user prompt contains:
        #
        # "Create 2-3 substantially different research strategies"

        if (
            "create 2-3 substantially different research strategies"
            in user
        ):
            return json.dumps(
                {
                    "strategies": [
                        {
                            "name": "Student Demand First",
                            "objective": (
                                "Understand student needs and "
                                "service gaps in Mumbai."
                            ),
                            "approach": (
                                "Research problems experienced by "
                                "Mumbai college students and identify "
                                "existing services addressing them."
                            ),
                            "research_questions": [
                                (
                                    "What needs and service problems "
                                    "do Mumbai college students "
                                    "commonly experience?"
                                ),
                                (
                                    "What student-focused services "
                                    "and businesses already operate "
                                    "around Mumbai colleges?"
                                ),
                            ],
                            "strengths": [
                                "Directly studies the target audience.",
                                "Can identify unmet service needs.",
                            ],
                            "weaknesses": [
                                (
                                    "Requires evidence from "
                                    "multiple sources."
                                )
                            ],
                        },
                        {
                            "name": "Market First",
                            "objective": (
                                "Understand the existing "
                                "student-focused market in Mumbai."
                            ),
                            "approach": (
                                "Research existing businesses, "
                                "services, and competitors serving "
                                "students."
                            ),
                            "research_questions": [
                                (
                                    "What student-focused businesses "
                                    "operate in Mumbai?"
                                ),
                                (
                                    "What services do existing "
                                    "student businesses provide?"
                                ),
                            ],
                            "strengths": [
                                "Shows existing market activity.",
                                "Can identify competitive gaps.",
                            ],
                            "weaknesses": [
                                (
                                    "Existing businesses do not "
                                    "necessarily prove strong "
                                    "student demand."
                                )
                            ],
                        },
                    ]
                }
            )

        # ====================================================
        # 2. STRATEGY SELECTOR
        # ====================================================

        # Actual selector user prompt contains:
        #
        # "Evaluate all strategies and select the one that should
        # receive the initial research budget."

        if (
            "evaluate all strategies and select the one" in user
            or "initial research budget" in user
        ):
            return json.dumps(
                {
                    "selected_strategy": "Student Demand First",
                    "evaluations": [
                        {
                            "strategy_name": "Student Demand First",
                            "information_value": 0.90,
                            "evidence_availability": 0.85,
                            "search_efficiency": 0.80,
                            "dead_end_risk": 0.20,
                            "reasoning": (
                                "It directly investigates the target "
                                "student population and their needs."
                            ),
                        },
                        {
                            "strategy_name": "Market First",
                            "information_value": 0.75,
                            "evidence_availability": 0.80,
                            "search_efficiency": 0.80,
                            "dead_end_risk": 0.30,
                            "reasoning": (
                                "It provides useful market information "
                                "but does not directly establish demand."
                            ),
                        },
                    ],
                    "reason": (
                        "Student Demand First provides the most "
                        "direct evidence about the target audience."
                    ),
                }
            )

        # ====================================================
        # 3. QUERY GENERATOR
        # ====================================================

        # Actual production system prompt starts with:
        #
        # "You are the search-query planner for an autonomous
        # research agent."

        if (
            "search-query planner" in system
            or "search-query planner for an autonomous research agent"
            in system
        ):
            return json.dumps(
                {
                    "query": (
                        "student services Mumbai college students"
                    ),
                    "reasoning": (
                        "The query preserves the Mumbai student "
                        "audience and focuses on student services "
                        "and needs."
                    ),
                }
            )

        # ====================================================
        # 4. EVIDENCE CRITIC
        # ====================================================

        if (
            "evidence critic" in system
            or "candidate evidence" in system
            or "assess the relevance" in system
            or "relevance_score" in system
            or "source_quality" in system
        ):
            return json.dumps(
                {
                    "assessments": [
                        {
                            "index": 0,
                            "relevant": True,
                            "relevance_score": 0.92,
                            "claim": (
                                "The evidence contains information "
                                "about student services in Mumbai."
                            ),
                            "source_quality": "medium",
                            "reasoning": (
                                "The result directly relates to "
                                "students and services in Mumbai."
                            ),
                        },
                        {
                            "index": 1,
                            "relevant": True,
                            "relevance_score": 0.88,
                            "claim": (
                                "The evidence discusses needs "
                                "experienced by college students "
                                "in Mumbai."
                            ),
                            "source_quality": "medium",
                            "reasoning": (
                                "The result is directly relevant "
                                "to the research question."
                            ),
                        },
                        {
                            "index": 2,
                            "relevant": True,
                            "relevance_score": 0.84,
                            "claim": (
                                "The evidence contains information "
                                "about student life and services."
                            ),
                            "source_quality": "medium",
                            "reasoning": (
                                "The result provides relevant "
                                "student-focused information."
                            ),
                        },
                    ]
                }
            )

        # ====================================================
        # 5. STRATEGY SWITCHER
        # ====================================================

        # Keep this AFTER query generation.
        #
        # The real StrategySwitcher should receive a decision
        # object, not a strategy set or query.

        if (
            "should_switch" in system
            and "target_strategy" in system
        ):
            return json.dumps(
                {
                    "should_switch": False,
                    "target_strategy": "",
                    "reason": (
                        "The current strategy remains appropriate "
                        "for the available evidence."
                    ),
                    "confidence": 0.90,
                }
            )

        # ====================================================
        # 6. FINAL SYNTHESIZER
        # ====================================================

        # The actual synthesizer receives the original mission,
        # strategy history, unresolved questions and evidence.

        if (
            "supplied research evidence" in user
            or (
                "original user mission" in user
                and "unresolved questions" in user
            )
        ):
            return json.dumps(
                {
                    "answer": (
                        "The available evidence indicates that "
                        "student-focused services are relevant in "
                        "Mumbai, but the current evidence is not "
                        "sufficient to identify one specific "
                        "business opportunity with confidence."
                    ),
                    "facts": [
                        (
                            "The research returned information "
                            "related to student services and "
                            "college students in Mumbai."
                        ),
                        (
                            "Multiple research results were "
                            "evaluated for relevance."
                        ),
                    ],
                    "inferences": [
                        (
                            "There may be opportunities to "
                            "investigate services addressing "
                            "student needs."
                        )
                    ],
                    "unknowns": [
                        (
                            "The current evidence does not "
                            "establish student willingness to pay."
                        ),
                        (
                            "The current evidence does not "
                            "establish market size or revenue "
                            "potential."
                        ),
                    ],
                    "unresolved_questions": [
                        (
                            "Which student problems have the "
                            "strongest unmet demand?"
                        )
                    ],
                    "sources": [],
                }
            )

        # ====================================================
        # FAIL LOUDLY
        # ====================================================

        raise AssertionError(
            "\n\n"
            "FakeLLM received an UNKNOWN PROMPT.\n\n"
            "SYSTEM PROMPT:\n"
            f"{system_prompt}\n\n"
            "USER PROMPT:\n"
            f"{user_prompt}\n"
        )


# ============================================================
# FAKE SERPAPI
# ============================================================

class FakeSerpApiTools:
    """
    Fake SerpApi implementation.

    No real HTTP requests.
    No SerpApi credits.
    """

    def web_search(
        self,
        query: str,
        location: str | None = None,
    ) -> dict:

        return {
            "organic_results": [
                {
                    "title": "Mumbai Student Services",
                    "link": (
                        "https://example.com/"
                        "mumbai-student-services"
                    ),
                    "snippet": (
                        "Information about student services "
                        "and support available to college "
                        "students in Mumbai."
                    ),
                },
                {
                    "title": "College Student Needs in Mumbai",
                    "link": (
                        "https://example.com/"
                        "college-student-needs"
                    ),
                    "snippet": (
                        "Information about common needs faced "
                        "by college students in Mumbai."
                    ),
                },
                {
                    "title": "Student Life in Mumbai",
                    "link": (
                        "https://example.com/"
                        "student-life-mumbai"
                    ),
                    "snippet": (
                        "Information about services and "
                        "activities relevant to students "
                        "in Mumbai."
                    ),
                },
            ]
        }

    def news_search(
        self,
        query: str,
    ) -> dict:

        return {
            "news_results": [
                {
                    "title": "Mumbai Student Services News",
                    "link": (
                        "https://example.com/"
                        "student-news"
                    ),
                    "snippet": (
                        "Recent information related to "
                        "student services in Mumbai."
                    ),
                    "source": "Example News",
                    "date": "2026-09-20",
                }
            ]
        }

    def maps_search(
        self,
        query: str,
    ) -> dict:

        return {
            "local_results": [
                {
                    "title": "Mumbai Student Hub",
                    "link": (
                        "https://example.com/"
                        "student-hub"
                    ),
                    "snippet": (
                        "Student-focused services operating "
                        "near Mumbai colleges."
                    ),
                    "address": "Mumbai, Maharashtra",
                }
            ]
        }

    def trends_search(
        self,
        query: str,
        geo: str = "IN",
    ) -> dict:

        return {
            "interest_over_time": {
                "timeline_data": [
                    {
                        "date": "2026-01",
                        "values": [
                            {"value": 50}
                        ],
                    },
                    {
                        "date": "2026-06",
                        "values": [
                            {"value": 70}
                        ],
                    },
                ]
            }
        }


# ============================================================
# COMPLETE END-TO-END TEST
# ============================================================

@pytest.mark.anyio
async def test_complete_research_loop():

    # --------------------------------------------------------
    # 1. FAKE DEPENDENCIES
    # --------------------------------------------------------

    llm = FakeLLM()

    serpapi_tools = FakeSerpApiTools()

    # Completely local test budget.
    # This does NOT consume real SerpApi credits.
    budget = SearchBudget(
        maximum=3
    )

    # --------------------------------------------------------
    # 2. RESEARCH EXECUTOR
    # --------------------------------------------------------

    executor = ResearchExecutor(
        serpapi_tools=serpapi_tools,
        llm=llm,
        budget=budget,
    )

    # --------------------------------------------------------
    # 3. STRATEGY ENGINE
    # --------------------------------------------------------

    strategy_engine = StrategyEngine(
        llm=llm
    )

    # --------------------------------------------------------
    # 4. STRATEGY SELECTOR
    # --------------------------------------------------------

    strategy_selector = StrategySelector(
        llm=llm
    )

    # --------------------------------------------------------
    # 5. STRATEGY SWITCHER
    # --------------------------------------------------------

    strategy_switcher = StrategySwitcher(
        llm=llm
    )

    # --------------------------------------------------------
    # 6. EVALUATOR
    # --------------------------------------------------------

    evaluator = EvidenceEvaluator()

    # --------------------------------------------------------
    # 7. SYNTHESIZER
    # --------------------------------------------------------

    synthesizer = FinalSynthesizer(
        llm=llm
    )

    # --------------------------------------------------------
    # 8. RESEARCH LOOP
    # --------------------------------------------------------

    loop = ResearchLoop(
        executor=executor,
        strategy_engine=strategy_engine,
        strategy_selector=strategy_selector,
        synthesizer=synthesizer,
        evaluator=evaluator,
        strategy_switcher=strategy_switcher,
    )

    # --------------------------------------------------------
    # 9. RUN MISSION
    # --------------------------------------------------------

    result = await loop.run(
        mission=MISSION,
        plan=(
            "Identify evidence-backed student-focused "
            "business opportunity areas in Mumbai."
        ),
    )

    # ========================================================
    # ASSERTIONS
    # ========================================================

    # --------------------------------------------------------
    # Mission
    # --------------------------------------------------------

    assert result is not None

    assert result.mission == MISSION

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    assert isinstance(
        result.answer,
        str,
    )

    assert result.answer.strip()

    # --------------------------------------------------------
    # Strategy
    # --------------------------------------------------------

    assert isinstance(
        result.strategy_history,
        list,
    )

    assert (
        "Student Demand First"
        in result.strategy_history
    )

    # --------------------------------------------------------
    # Evaluations
    # --------------------------------------------------------

    assert isinstance(
        result.evaluations,
        list,
    )

    assert len(result.evaluations) >= 1

    # --------------------------------------------------------
    # Searches
    # --------------------------------------------------------

    assert result.searches_used >= 1

    assert (
        result.searches_used
        <= budget.maximum
    )

    assert result.remaining_searches >= 0

    assert budget.used <= budget.maximum

    # --------------------------------------------------------
    # Structured synthesis
    # --------------------------------------------------------

    assert isinstance(
        result.facts,
        list,
    )

    assert isinstance(
        result.inferences,
        list,
    )

    assert isinstance(
        result.unknowns,
        list,
    )

    assert isinstance(
        result.unresolved_questions,
        list,
    )

    # --------------------------------------------------------
    # Minimum useful result
    # --------------------------------------------------------

    assert len(result.facts) >= 1

    assert len(result.unknowns) >= 1

    # --------------------------------------------------------
    # Source safety
    # --------------------------------------------------------

    if hasattr(result, "sources"):

        valid_urls = {
            "https://example.com/mumbai-student-services",
            "https://example.com/college-student-needs",
            "https://example.com/student-life-mumbai",
            "https://example.com/student-news",
            "https://example.com/student-hub",
        }

        for source in result.sources:
            assert source.url in valid_urls

    # --------------------------------------------------------
    # Budget safety
    # --------------------------------------------------------

    assert budget.used <= 3

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("MISSIONCTRL RESEARCH LOOP")
    print("=" * 70)

    print()
    print("MISSION:")
    print(result.mission)

    print()
    print("ANSWER:")
    print(result.answer)

    print()
    print("STRATEGY HISTORY:")

    for strategy in result.strategy_history:
        print(f"  - {strategy}")

    print()
    print("SEARCHES USED:")
    print(result.searches_used)

    print()
    print("REMAINING SEARCHES:")
    print(result.remaining_searches)

    print()
    print("FACTS:")

    for fact in result.facts:
        print(f"  - {fact}")

    print()
    print("INFERENCES:")

    for inference in result.inferences:
        print(f"  - {inference}")

    print()
    print("UNKNOWNS:")

    for unknown in result.unknowns:
        print(f"  - {unknown}")

    print()
    print("UNRESOLVED QUESTIONS:")

    for question in result.unresolved_questions:
        print(f"  - {question}")

    print()
    print("=" * 70)
    print("MISSIONCTRL END-TO-END TEST PASSED")
    print("=" * 70)