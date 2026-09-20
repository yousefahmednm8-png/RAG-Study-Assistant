import requests
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