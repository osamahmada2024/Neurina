import requests
from ...config import settings

def ask_ollama(model: str, prompt: str) -> str:
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7
    }
    resp = requests.post(url, json=payload, timeout=120)

    resp.raise_for_status()

    return resp.json().get("response", "").strip()
