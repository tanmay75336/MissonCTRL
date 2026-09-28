from unittest.mock import AsyncMock, patch

import pytest

from app.llm.factory import get_llm_provider


@pytest.mark.parametrize(
    "name",
    [
        "groq",
        "gemini",
        "nemotron",
    ],
)
def test_provider_factory(name: str):
    """
    Verify that the factory maps each supported provider
    name to the correct provider implementation.

    This test does NOT make real API calls.
    """

    provider = get_llm_provider(name)

    assert provider is not None
    assert provider.__class__.__name__ in {
        "GroqProvider",
        "GeminiProvider",
        "NemotronProvider",
    }


@pytest.mark.anyio
async def test_groq_provider_generate():
    """
    Unit-test the Groq provider without contacting Groq.
    """

    fake_response = "AI agents need tools to interact with external systems."

    with patch(
        "app.llm.groq.AsyncOpenAI"
    ) as mock_client:

        mock_client.return_value.chat.completions.create = (
            AsyncMock(
                return_value=type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": fake_response
                                        },
                                    )()
                                },
                            )()
                        ]
                    },
                )()
            )
        )

        from app.llm.groq import GroqProvider

        provider = GroqProvider()

        result = await provider.generate(
            system_prompt="You are MISSIONCTRL.",
            user_prompt="Explain why agents need tools.",
        )

        assert result == fake_response