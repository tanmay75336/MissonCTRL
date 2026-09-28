from pydantic import BaseModel


class ToolDecision(BaseModel):
    tool: str
    query: str
    reason: str


class ToolRouter:
    """
    Routes a research question to the most appropriate
    SerpApi tool.

    The router uses contextual rules rather than treating
    individual words as definitive signals.
    """

    VALID_TOOLS = {
        "web_search",
        "news_search",
        "maps_search",
        "trends_search",
    }

    def route(
        self,
        question: str,
        context: str = "",
    ) -> ToolDecision:

        q = question.lower().strip()

        # --------------------------------
        # TRENDS
        # --------------------------------

        trend_phrases = [
            "trend",
            "trends",
            "trending",
            "popularity",
            "growing demand",
            "growth in demand",
            "demand over time",
            "change over time",
            "changed over time",
            "search interest",
            "rising demand",
            "declining demand",
        ]

        if any(
            phrase in q
            for phrase in trend_phrases
        ):
            return ToolDecision(
                tool="trends_search",
                query=question,
                reason=(
                    "The question requires trend, "
                    "demand, popularity, or search-interest data."
                ),
            )

        # --------------------------------
        # MAPS
        # --------------------------------

        maps_phrases = [
            "near ",
            "nearby",
            "in my area",
            "local businesses",
            "physical businesses",
            "businesses near",
            "competitors near",
            "locations of",
            "cafes near",
            "shops near",
            "stores near",
            "colleges near",
            "gyms near",
        ]

        if any(
            phrase in q
            for phrase in maps_phrases
        ):
            return ToolDecision(
                tool="maps_search",
                query=question,
                reason=(
                    "The question requires local "
                    "business or geographic information."
                ),
            )

        # --------------------------------
        # NEWS
        # --------------------------------
        #
        # Important:
        # "launched" by itself is NOT a news signal.
        # It may mean "a business that can be launched".

        news_phrases = [
            "latest news",
            "recent news",
            "news about",
            "recent announcement",
            "recent announcements",
            "announced recently",
            "latest update",
            "recent update",
            "today's news",
            "this week's news",
            "this month's news",
        ]

        if any(
            phrase in q
            for phrase in news_phrases
        ):
            return ToolDecision(
                tool="news_search",
                query=question,
                reason=(
                    "The question explicitly requires "
                    "recent news or announcements."
                ),
            )

        # --------------------------------
        # DEFAULT → WEB
        # --------------------------------

        return ToolDecision(
            tool="web_search",
            query=question,
            reason=(
                "General web research is the most "
                "appropriate tool."
            ),
        )