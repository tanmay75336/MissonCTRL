from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.agent.budget import SearchBudget
from app.agent.evidence_critic import EvidenceCritic
from app.agent.query_generator import QueryGenerator
from app.agent.tool_router import ToolDecision, ToolRouter
from app.llm.base import LLMProvider
from app.tools.serpapi import SerpApiTools


class Evidence(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    published_at: str | None = None

    tool: str
    query: str

    # Evidence Critic fields
    relevant: bool = True
    relevance_score: float = 0.0
    claim: str | None = None
    source_quality: str = "unknown"
    critic_reasoning: str = ""
    objective_id: str = ""
    source_classification: str = "GENERAL"

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ResearchResult(BaseModel):
    question: str

    decision: ToolDecision

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    raw_evidence_count: int = 0
    filtered_evidence_count: int = 0

    search_number: int
    remaining_searches: int
    objective_id: str = ""


class ResearchExecutor:
    """
    Executes research questions through SerpApi.

    Pipeline:

        Question
            ↓
        Tool Router
            ↓
        Query Generator
            ↓
        Focused Search Query
            ↓
        SerpApi
            ↓
        Normalize Results
            ↓
        Evidence Critic
            ↓
        Filtered Evidence
            ↓
        Research Evaluator
    """

    # Maximum number of SerpApi results sent to the LLM critic.
    # This prevents one search from creating dozens of LLM calls.
    MAX_CRITIC_CANDIDATES = 10

    def __init__(
        self,
        serpapi_tools: SerpApiTools,
        llm: LLMProvider,
        router: ToolRouter | None = None,
        budget: SearchBudget | None = None,
        evidence_critic: EvidenceCritic | None = None,
    ):

        self.serpapi = serpapi_tools

        self.llm = llm

        self.router = (
            router
            or ToolRouter()
        )

        self.query_generator = QueryGenerator(
            llm=llm
        )

        self.evidence_critic = (
            evidence_critic
            or EvidenceCritic(llm)
        )

        self.budget = (
            budget
            or SearchBudget(
                maximum=8
            )
        )

    async def research(
        self,
        question: str,
        context: str = "",
        previous_queries: set[str] | None = None,
        objective_id: str = "",
    ) -> ResearchResult:

        # --------------------------------
        # 1. ROUTE QUESTION
        # --------------------------------

        decision = self.router.route(
            question=question,
            context=context,
        )

        # --------------------------------
        # 2. GENERATE SEARCH QUERY
        # --------------------------------

        generated_query = (
            await self.query_generator.generate(
                mission=context,
                question=question,
                tool=decision.tool,
            )
        )

        # Replace the raw research question
        # with the focused search query.

        decision = ToolDecision(
            tool=decision.tool,
            query=generated_query.query,
            reason=(
                f"{decision.reason} "
                "Generated focused search query."
            ),
        )

        normalized_query = self._normalize_search_text(
            decision.query
        )

        if (
            previous_queries is not None
            and normalized_query in previous_queries
        ):
            raise RuntimeError(
                "MISSIONCTRL prevented a duplicate search query."
            )

        print(
            f"[ROUTER] Tool: {decision.tool}"
        )

        print(
            f"[QUERY] {decision.query}"
        )

        # --------------------------------
        # 3. CHECK SEARCH BUDGET
        # --------------------------------

        if self.budget.exhausted:

            raise RuntimeError(
                "MISSIONCTRL search budget exhausted. "
                f"Maximum searches: "
                f"{self.budget.maximum}"
            )

        # --------------------------------
        # 4. EXECUTE SERPAPI SEARCH
        # --------------------------------

        raw_results = self._execute_tool(
            decision
        )

        # --------------------------------
        # 5. COUNT SEARCH
        # --------------------------------

        self.budget.consume()

        # --------------------------------
        # 6. NORMALIZE SEARCH RESULTS
        # --------------------------------

        evidence = self._normalize_results(
            raw_results=raw_results,
            decision=decision,
        )
        for item in evidence:
            item.objective_id = objective_id

        raw_evidence_count = len(
            evidence
        )

        print(
            f"[SEARCH] Raw evidence: "
            f"{raw_evidence_count}"
        )

        # --------------------------------
        # 7. REMOVE DUPLICATE URLS
        # --------------------------------

        evidence = self._deduplicate_evidence(
            evidence
        )

        print(
            f"[SEARCH] After deduplication: "
            f"{len(evidence)}"
        )

        # --------------------------------
        # 8. SELECT TOP CANDIDATES
        # --------------------------------
        #
        # We don't send every SerpApi result
        # to the LLM.
        #
        # This controls Groq usage and latency.

        candidates = evidence[
            : self.MAX_CRITIC_CANDIDATES
        ]

        print(
            f"[CRITIC] Evaluating "
            f"{len(candidates)} candidates..."
        )

        # --------------------------------
        # 9. EVIDENCE CRITIC
        # --------------------------------

        if candidates:

            assessments = (
                await self.evidence_critic.assess_many(
                    question=question,
                    evidence_items=candidates,
                )
            )

        else:

            assessments = []

        # --------------------------------
        # 10. APPLY CRITIC ASSESSMENTS
        # --------------------------------

        filtered_evidence = []

        for assessment in assessments:

            # Protect against invalid LLM indexes.

            if (
                assessment.index < 0
                or assessment.index >= len(candidates)
            ):
                continue

            item = candidates[
                assessment.index
            ]

            # Store critic information
            # directly on the evidence object.

            item.relevant = (
                assessment.relevant
            )

            item.relevance_score = (
                assessment.relevance_score
            )

            item.claim = (
                assessment.claim
            )

            item.source_quality = (
                assessment.source_quality
            )

            item.critic_reasoning = (
                assessment.reasoning
            )

            # --------------------------------
            # KEEP / REJECT
            # --------------------------------

            if (
                assessment.relevant
                and assessment.relevance_score >= 0.60
            ):

                filtered_evidence.append(
                    item
                )

        print(
            f"[CRITIC] Kept "
            f"{len(filtered_evidence)} / "
            f"{len(candidates)} "
            f"relevant results."
        )

        # --------------------------------
        # 11. RETURN FILTERED EVIDENCE
        # --------------------------------

        return ResearchResult(
            question=question,
            decision=decision,
            evidence=filtered_evidence,
            raw_evidence_count=raw_evidence_count,
            filtered_evidence_count=len(
                filtered_evidence
            ),
            search_number=self.budget.used,
            remaining_searches=self.budget.remaining,
            objective_id=objective_id,
        )

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        """Normalize a query for conservative duplicate detection."""

        return " ".join(
            "".join(
                character if character.isalnum() else " "
                for character in text.lower()
            ).split()
        )

    # ====================================
    # SERPAPI TOOL EXECUTION
    # ====================================

    def _execute_tool(
        self,
        decision: ToolDecision,
    ) -> dict[str, Any]:

        tool = decision.tool

        query = decision.query

        if tool == "web_search":

            return self.serpapi.web_search(
                query
            )

        if tool == "news_search":

            return self.serpapi.news_search(
                query
            )

        if tool == "maps_search":

            return self.serpapi.maps_search(
                query
            )

        if tool == "trends_search":

            return self.serpapi.trends_search(
                query
            )

        raise ValueError(
            f"Unsupported SerpApi tool: {tool}"
        )

    # ====================================
    # NORMALIZE SERPAPI RESULTS
    # ====================================

    def _normalize_results(
        self,
        raw_results: dict[str, Any],
        decision: ToolDecision,
    ) -> list[Evidence]:

        evidence: list[Evidence] = []

        # --------------------------------
        # GENERAL WEB RESULTS
        # --------------------------------

        organic_results = raw_results.get(
            "organic_results",
            [],
        )

        for result in organic_results:

            url = result.get(
                "link",
                "",
            )

            # Google organic results normally provide a
            # displayed_link rather than a dedicated source field.
            # Evidence requires a source, so preserve an explicit
            # source when present and otherwise derive one from the
            # result URL.
            source_value = (
                result.get("source")
                or result.get("displayed_link")
                or urlparse(url).netloc
            )

            if isinstance(source_value, dict):
                source = source_value.get("name", "")
            else:
                source = str(source_value)

            evidence.append(
                Evidence(
                    title=result.get(
                        "title",
                        "",
                    ),
                    url=url,
                    snippet=result.get(
                        "snippet",
                        "",
                    ),
                    source=source,
                    published_at=result.get(
                        "date",
                        result.get(
                            "published_at"
                        ),
                    ),
                    tool=decision.tool,
                    query=decision.query,
                    metadata={
                        "position": result.get(
                            "position"
                        ),
                        "displayed_link": result.get(
                            "displayed_link"
                        ),
                    },
                )
            )

        # --------------------------------
        # NEWS RESULTS
        # --------------------------------

        news_results = raw_results.get(
            "news_results",
            [],
        )

        for result in news_results:

            source_data = result.get(
                "source",
                {},
            )

            if isinstance(
                source_data,
                dict,
            ):

                source = source_data.get(
                    "name",
                    "",
                )

            else:

                source = str(
                    source_data
                )

            evidence.append(
                Evidence(
                    title=result.get(
                        "title",
                        "",
                    ),
                    url=result.get(
                        "link",
                        "",
                    ),
                    snippet=result.get(
                        "snippet",
                        "",
                    ),
                    source=source,
                    published_at=result.get(
                        "date",
                        result.get(
                            "published_at"
                        ),
                    ),
                    tool=decision.tool,
                    query=decision.query,
                    metadata={
                        "position": result.get(
                            "position"
                        ),
                    },
                )
            )

        # --------------------------------
        # MAPS RESULTS
        # --------------------------------

        local_results = raw_results.get(
            "local_results",
            [],
        )

        for result in local_results:

            address = result.get(
                "address",
                "",
            )

            evidence.append(
                Evidence(
                    title=result.get(
                        "title",
                        result.get(
                            "name",
                            "",
                        ),
                    ),
                    url=result.get(
                        "link",
                        result.get(
                            "website",
                            "",
                        ),
                    ),
                    snippet=(
                        f"Address: {address}"
                        if address
                        else ""
                    ),
                    source="Google Maps",
                    tool=decision.tool,
                    query=decision.query,
                    metadata={
                        "rating": result.get(
                            "rating"
                        ),
                        "reviews": result.get(
                            "reviews"
                        ),
                        "type": result.get(
                            "type"
                        ),
                    },
                )
            )

        # --------------------------------
        # GENERIC FALLBACK
        # --------------------------------

        if not evidence:

            evidence.extend(
                self._extract_generic_results(
                    raw_results,
                    decision,
                )
            )

        return evidence

    # ====================================
    # DEDUPLICATION
    # ====================================

    def _deduplicate_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:

        unique: list[Evidence] = []

        seen_urls: set[str] = set()

        for item in evidence:

            url = item.url.strip()

            # If URL exists, use it as the
            # primary duplicate identifier.

            if url:

                normalized_url = (
                    url.rstrip("/")
                    .lower()
                )

                if normalized_url in seen_urls:
                    continue

                seen_urls.add(
                    normalized_url
                )

            else:

                # Some results may not have URLs.
                # Use title + source as fallback.

                fallback_key = (
                    f"{item.title.strip().lower()}|"
                    f"{item.source.strip().lower()}"
                )

                if fallback_key in seen_urls:
                    continue

                seen_urls.add(
                    fallback_key
                )

            unique.append(item)

        return unique

    # ====================================
    # GENERIC RESULT EXTRACTION
    # ====================================

    def _extract_generic_results(
        self,
        raw_results: dict[str, Any],
        decision: ToolDecision,
    ) -> list[Evidence]:

        evidence: list[Evidence] = []

        for key, value in raw_results.items():

            if not isinstance(
                value,
                list,
            ):
                continue

            for item in value:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                title = (
                    item.get("title")
                    or item.get("name")
                    or ""
                )

                url = (
                    item.get("link")
                    or item.get("url")
                    or ""
                )

                snippet = (
                    item.get("snippet")
                    or item.get("description")
                    or ""
                )

                if not title and not snippet:
                    continue

                evidence.append(
                    Evidence(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=str(key),
                        tool=decision.tool,
                        query=decision.query,
                        metadata=item,
                    )
                )

        return evidence
