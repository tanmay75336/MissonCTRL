from openai import AsyncOpenAI

from app.config import settings
from app.llm.base import LLMProvider


class NemotronProvider(LLMProvider):

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        response = await self.client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
        )

        return response.choices[0].message.content