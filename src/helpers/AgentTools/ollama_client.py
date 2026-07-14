import requests
from ...config import settings


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot produce a usable response."""


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout_seconds: int = 120):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, model: str, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            response = resp.json().get("response", "").strip()
        except requests.exceptions.RequestException as exc:
            raise OllamaClientError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaClientError("Ollama returned invalid JSON") from exc

        if not response:
            raise OllamaClientError("Ollama returned an empty response")
        return response


_default_client: OllamaClient | None = None


def get_default_ollama_client() -> OllamaClient:
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient()
    return _default_client


def ask_ollama(
    model: str,
    prompt: str,
    client: OllamaClient | None = None,
) -> str:
    return (client or get_default_ollama_client()).generate(model, prompt)
