import logging
import requests

logger = logging.getLogger(__name__)


class PerplexityAsyncClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def create_async_job(self, query: str) -> dict:
        payload = {
            "request": {
                "model": "sonar-deep-research",
                "messages": [
                    {"role": "system", "content": "You are a thorough research assistant."},
                    {"role": "user", "content": query},
                ],
                "reasoning_effort": "medium",
                "web_search_options": {"search_context_size": "high"},
                "search_mode": "web",
            }
        }
        res = requests.post(
            f"{self.base_url}/async/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        res.raise_for_status()
        return res.json()

    def check_job_status(self, request_id: str) -> dict:
        res = requests.get(
            f"{self.base_url}/async/chat/completions/{request_id}",
            headers=self.headers,
            timeout=30,
        )
        res.raise_for_status()
        return res.json()
