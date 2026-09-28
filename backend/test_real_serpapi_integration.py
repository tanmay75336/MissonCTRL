"""Opt-in live SerpApi integration coverage.

Run only when deliberately requested:
    $env:RUN_REAL_SERPAPI_INTEGRATION = "1"
    python -m pytest -v -s test_real_serpapi_integration.py
"""

import os

import pytest

from app.agent.budget import SearchBudget
from app.agent.research_executor import ResearchExecutor
from app.config import settings
from app.llm.groq import GroqProvider
from app.tools.serpapi import SerpApiTools


pytestmark = pytest.mark.integration

MISSION = "Find current information about the SerpApi India Hackathon 2026."


@pytest.mark.anyio
@pytest.mark.skipif(
    os.getenv("RUN_REAL_SERPAPI_INTEGRATION") != "1",
    reason="Set RUN_REAL_SERPAPI_INTEGRATION=1 to authorize one live SerpApi search.",
)
async def test_real_serpapi_integration() -> None:
    """Make exactly one Google search and validate its Evidence conversion."""

    # These presence checks intentionally never display secret values.
    assert settings.serpapi_api_key
    assert settings.groq_api_key
    assert settings.default_llm.lower() == "groq"

    budget = SearchBudget(maximum=1)
    executor = ResearchExecutor(
        serpapi_tools=SerpApiTools(),
        llm=GroqProvider(),
        budget=budget,
    )

    result = await executor.research(
        question=MISSION,
        context=MISSION,
    )

    assert result.decision.tool == "web_search"
    assert result.raw_evidence_count >= 1
    assert result.evidence
    assert any(
        item.title.strip()
        and item.url.strip()
        and item.snippet.strip()
        and item.source.strip()
        for item in result.evidence
    )
    assert budget.used == 1
    assert budget.remaining == 0
    assert result.search_number == 1
    assert result.remaining_searches == 0

    print("## REAL SERPAPI INTEGRATION")
    print("Authentication: PASS")
    print("Search execution: PASS")
    print(f"Results returned: {result.raw_evidence_count}")
    print("Evidence normalization: PASS")
    print("Searches used: 1")
    print("Remaining budget: 0")
    print("API key exposed: NO")
