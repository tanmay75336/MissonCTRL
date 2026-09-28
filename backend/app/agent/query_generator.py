import json
import re

from pydantic import BaseModel

from app.llm.base import LLMProvider


class SearchQuery(BaseModel):
    query: str
    reasoning: str


class QueryGenerator:
    """
    Converts a research question into one precise search-engine query.

    Important safety property:
    The generated query must not introduce unsupported facts,
    years, dates, statistics, budgets, locations, or constraints.
    """

    SYSTEM_PROMPT = """
You are the search-query planner for an autonomous research agent.

Your job is to convert ONE research question into ONE precise
search-engine query.

============================================================
CORE RULE — ZERO INVENTION
============================================================

The generated query may ONLY use facts, constraints, entities,
locations, audiences, budgets, dates, years, quantities, and
requirements that are explicitly present in either:

1. MISSION
2. RESEARCH QUESTION

NEVER invent or introduce:

- years
- dates
- statistics
- prices
- budgets
- percentages
- market sizes
- locations
- organizations
- demographics
- time periods
- regulations
- facts
- constraints
- assumptions
- specific products
- specific companies

unless they are explicitly present in the mission or
research question.

============================================================
IMPORTANT DATE RULE
============================================================

If the mission or research question says:

- latest
- recent
- current
- currently
- today
- this month
- this year

DO NOT replace that wording with a specific year or date.

For example:

BAD:
"latest student business trends Mumbai 2024"

when "2024" was never provided.

GOOD:
"latest student business trends Mumbai"

============================================================
QUERY REQUIREMENTS
============================================================

The query must:

- preserve important mission constraints
- preserve explicitly stated location
- preserve explicitly stated target audience
- preserve explicitly stated budget
- preserve explicitly stated dates or years
- focus directly on the research question
- remove conversational wording
- be concise
- be suitable for the selected search tool
- NOT answer the research question
- NOT add assumptions
- NOT add unsupported specificity
- NOT add unrelated entities

============================================================
REASONING REQUIREMENTS
============================================================

The reasoning must briefly explain how the query was formed.

Do not invent information in the reasoning either.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

{
    "query": "string",
    "reasoning": "string"
}
"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def generate(
        self,
        mission: str,
        question: str,
        tool: str,
    ) -> SearchQuery:

        user_prompt = f"""
MISSION:
{mission}

RESEARCH QUESTION:
{question}

SELECTED SEARCH TOOL:
{tool}

Generate ONE focused search query appropriate for the
selected search tool.

Remember:

- Use only information explicitly present above.
- Do not invent a year.
- Do not invent a date.
- Do not invent statistics.
- Do not invent a budget.
- Do not invent a location.
- Do not invent a demographic.
- Do not invent a constraint.
- Do not answer the question.

Return ONLY valid JSON.
"""

        response = await self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # ---------------------------------------------------------
        # 1. PARSE JSON
        # ---------------------------------------------------------

        try:
            data = json.loads(response)

        except json.JSONDecodeError as error:
            raise ValueError(
                "Query generator returned invalid JSON.\n"
                f"Response: {response}"
            ) from error

        # ---------------------------------------------------------
        # 2. VALIDATE SCHEMA
        # ---------------------------------------------------------

        try:
            result = SearchQuery.model_validate(data)

        except Exception as error:
            raise ValueError(
                "Query generator returned invalid SearchQuery structure.\n"
                f"Response: {response}"
            ) from error

        # ---------------------------------------------------------
        # 3. BASIC VALIDATION
        # ---------------------------------------------------------

        if not result.query.strip():
            raise ValueError(
                "Query generator returned an empty query."
            )

        if not result.reasoning.strip():
            raise ValueError(
                "Query generator returned empty reasoning."
            )

        # ---------------------------------------------------------
        # 4. SECURITY / GROUNDING VALIDATION
        # ---------------------------------------------------------

        self._validate_query(
            query=result.query,
            mission=mission,
            question=question,
        )

        return result

    @staticmethod
    def _validate_query(
        query: str,
        mission: str,
        question: str,
    ) -> None:
        """
        Programmatic guard against obvious hallucinated
        temporal/numeric information.

        The LLM prompt is the first layer.
        This validator is the second layer.
        """

        source_text = (
            f"{mission} {question}"
        ).lower()

        query_text = query.lower()

        # ---------------------------------------------------------
        # A. YEAR VALIDATION
        # ---------------------------------------------------------

        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            query,
        )

        for year in years:

            if year not in source_text:

                raise ValueError(
                    "Query generator invented an unsupported year: "
                    f"{year}\n"
                    f"Generated query: {query}"
                )

        # ---------------------------------------------------------
        # B. PERCENTAGE VALIDATION
        # ---------------------------------------------------------

        percentages = re.findall(
            r"\b\d+(?:\.\d+)?\s*%",
            query,
        )

        for percentage in percentages:

            if percentage.lower() not in source_text:

                raise ValueError(
                    "Query generator invented an unsupported "
                    f"percentage: {percentage}\n"
                    f"Generated query: {query}"
                )

        # ---------------------------------------------------------
        # C. CURRENCY / NUMERIC VALIDATION
        # ---------------------------------------------------------

        # Detect common currency expressions such as:
        # ₹2 lakh
        # ₹20,000
        # $500
        # 5000 INR
        # 2 lakh
        numeric_patterns = [
            r"₹\s*[\d,]+(?:\.\d+)?",
            r"\$\s*[\d,]+(?:\.\d+)?",
            r"\b\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores)\b",
            r"\b\d[\d,]*(?:\.\d+)?\s*(?:inr|usd|eur|gbp)\b",
        ]

        for pattern in numeric_patterns:

            values = re.findall(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            for value in values:

                normalized_value = value.lower()

                if normalized_value not in source_text:

                    raise ValueError(
                        "Query generator introduced an unsupported "
                        f"numeric/currency constraint: {value}\n"
                        f"Generated query: {query}"
                    )

        # ---------------------------------------------------------
        # D. EMPTY / EXCESSIVE QUERY CHECK
        # ---------------------------------------------------------

        if len(query.strip()) < 3:

            raise ValueError(
                "Generated search query is too short."
            )

        if len(query.split()) > 30:

            raise ValueError(
                "Generated search query is unnecessarily long."
            )

        # ---------------------------------------------------------
        # E. FINAL SANITY CHECK
        # ---------------------------------------------------------

        if not query_text.strip():

            raise ValueError(
                "Generated search query is empty."
            )