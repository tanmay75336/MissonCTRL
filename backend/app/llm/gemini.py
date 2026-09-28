import asyncio

from google import genai
from google.genai import errors

from app.config import settings
from app.llm.base import LLMProvider


class GeminiProvider(LLMProvider):

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=user_prompt,
                    config={
                        "system_instruction": system_prompt,
                        "thinking_config": {
                            "thinking_level": "low"
                        },
                    },
                )

                return response.text

            except errors.ClientError as error:

                # 429 = quota exhausted.
                # Retrying immediately is pointless.
                if error.code == 429:
                    raise RuntimeError(
                        "Gemini quota exhausted. "
                        "Use another LLM provider."
                    ) from error

                raise

            except errors.ServerError as error:

                # 503 = temporary server availability problem.
                if error.code == 503:

                    if attempt == max_retries - 1:
                        raise

                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time}s..."
                    )

                    await asyncio.sleep(wait_time)

                else:
                    raise