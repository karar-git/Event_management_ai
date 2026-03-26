"""
Fal.ai OpenRouter Client for LLM API calls
"""

import httpx
from typing import Optional, AsyncGenerator
from config import config


class FalAIClient:
    """Client for interacting with Fal.ai OpenRouter API"""

    BASE_URL = "https://queue.fal.run/openrouter/router"

    def __init__(self):
        self.api_key = config.FAL_KEY
        self.default_model = config.DEFAULT_MODEL

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate a response from the LLM

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            model: Model to use (defaults to config default)
            temperature: Creativity setting (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            dict with 'output' and 'usage' fields
        """
        payload = {
            "prompt": prompt,
            "model": model or self.default_model,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.BASE_URL, headers=self._get_headers(), json=payload
            )
            response.raise_for_status()
            return response.json()

    async def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the LLM

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            model: Model to use
            temperature: Creativity setting

        Yields:
            str: Chunks of the response
        """
        payload = {
            "prompt": prompt,
            "model": model or self.default_model,
            "temperature": temperature,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        stream_url = self.BASE_URL.replace("queue.fal.run", "fal.run")

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", stream_url, headers=self._get_headers(), json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    if chunk:
                        yield chunk


# Global client instance
fal_client = FalAIClient()
