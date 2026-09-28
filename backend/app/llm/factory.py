from app.config import settings

from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.nemotron import NemotronProvider


def get_llm_provider(provider: str | None = None):

    provider = provider or settings.default_llm

    providers = {
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "nemotron": NemotronProvider,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown LLM provider: {provider}"
        )

    return providers[provider]()