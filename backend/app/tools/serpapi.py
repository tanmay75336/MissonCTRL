from typing import Any

import serpapi

from app.config import settings


class SerpApiTools:

    def __init__(self):
        api_key = settings.serpapi_api_key

        if not api_key:
            raise RuntimeError(
                "SERPAPI_API_KEY is not configured."
            )

        self.client = serpapi.Client(
            api_key=api_key
        )

    def web_search(
        self,
        query: str,
        location: str | None = None,
    ) -> dict[str, Any]:

        params = {
            "engine": "google",
            "q": query,
        }

        if location:
            params["location"] = location

        return self.client.search(params)

    def news_search(
        self,
        query: str,
    ) -> dict[str, Any]:

        return self.client.search({
            "engine": "google_news",
            "q": query,
        })

    def maps_search(
        self,
        query: str,
    ) -> dict[str, Any]:

        return self.client.search({
            "engine": "google_maps",
            "q": query,
        })

    def trends_search(
        self,
        query: str,
        geo: str = "IN",
    ) -> dict[str, Any]:

        return self.client.search({
            "engine": "google_trends",
            "q": query,
            "geo": geo,
        })