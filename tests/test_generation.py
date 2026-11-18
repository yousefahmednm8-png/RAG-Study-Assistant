import pytest
import requests

from src.generation import Generator


def test_generate_parses_ollama_response(monkeypatch):

    captured = {}

    class FakeResponse:

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "role": "assistant",
                    "content": "The answer.",
                }
            }

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    generator = Generator(
        model_name="fake:latest",
        base_url="http://localhost:11434",
    )

    answer = generator.generate("hello")

    assert answer == "The answer."

    assert captured["url"] == (
        "http://localhost:11434/api/chat"
    )

    assert captured["payload"]["model"] == "fake:latest"

    assert captured["payload"]["stream"] is False

    assert captured["timeout"] is not None


def test_generate_raises_on_http_error(monkeypatch):

    class BadResponse:

        def raise_for_status(self):
            raise requests.HTTPError("boom")

    def fake_post(url, json=None, timeout=None):
        return BadResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    generator = Generator(
        model_name="fake:latest",
        base_url="http://localhost:11434",
    )

    with pytest.raises(requests.HTTPError):

        generator.generate("hello")