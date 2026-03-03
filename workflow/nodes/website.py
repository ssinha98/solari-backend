import logging
from app import scrape_website_structured

logger = logging.getLogger(__name__)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Website node executor.

    Scrapes a URL with Firecrawl using structured JSON extraction (fixed
    schema: success + answer). Reads url and prompt from node config
    (after interpolation). Sets output to the extracted answer string.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Website")

    url = (node_data.get("websiteUrl") or node_data.get("url") or "").strip()
    prompt = (node_data.get("websitePrompt") or node_data.get("prompt") or "").strip()

    if not url:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": "Website node has no 'url' / 'websiteUrl' configured",
        }
    if not prompt:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": "Website node has no 'prompt' / 'websitePrompt' configured",
        }

    logger.info(f"Website node '{label}' scraping {url} with structured prompt")

    try:
        raw = scrape_website_structured(url=url, prompt=prompt)
        data = raw.get("data") or {}
        json_val = data.get("json") if isinstance(data, dict) else getattr(data, "json", None)
        if isinstance(json_val, dict):
            answer = json_val.get("answer")
        else:
            answer = getattr(json_val, "answer", None) if json_val is not None else None
        if answer is None:
            return {
                "output": None,
                "outputVariable": node_data.get("outputVariable"),
                "branch": None,
                "pause": False,
                "error": "No answer in website response",
            }
        return {
            "output": answer,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": None,
        }
    except Exception as e:
        logger.exception(f"Website node '{label}' failed")
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": str(e),
        }
