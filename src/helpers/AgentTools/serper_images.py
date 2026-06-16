import requests
import time
from typing import List
from ...config import settings


def search_images(query: str, num_results: int = 10, retry_count: int = 3) -> List[str]:
    for attempt in range(retry_count):
        try:
            resp = requests.get(
                "https://google.serper.dev/images",
                headers={"X-API-KEY": settings.Search_Secret_API_KEY},
                params={"q": query, "num": num_results},
                timeout=30
            )
            resp.raise_for_status()

            images = resp.json().get("images", [])
            image_urls = [img.get("imageUrl") for img in images if img.get("imageUrl")]

            return image_urls

        except requests.exceptions.Timeout as e:
            print(f"Serper API timeout (attempt {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            continue

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:  # Rate limited
                print(f"Serper API rate limited (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(5 * (2 ** attempt))  # Exponential backoff, longer for rate limit
                continue
            else:
                print(f"Serper API HTTP error (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                continue

        except Exception as e:
            print(f"Serper API error (attempt {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
            continue

    # All retries exhausted
    print(f"Failed to search images after {retry_count} attempts")
    return []
