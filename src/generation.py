import os
import requests

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None


class Generator:

    def __init__(
        self,
        model_name: str = "gemma4:31b-cloud",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
            },
            timeout=300,
        )

        response.raise_for_status()
        return response.json()["message"]["content"]


class GeminiGenerator:
    """Generator backed by the Google Gemini API."""

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        api_key: str | None = None,
    ) -> None:
        if genai is None:
            raise RuntimeError("google-genai is not installed.")
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is missing.")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text.strip()