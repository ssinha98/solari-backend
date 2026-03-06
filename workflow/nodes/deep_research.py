import logging
import os
import requests
from openai import OpenAI
from firebase_config import firestore
from workflow.deep_research.perplexity import PerplexityAsyncClient

logger = logging.getLogger(__name__)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _research_openai(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """Start OpenAI o3-deep-research async job, store thread id on run, pause for webhook to complete."""
    node_data = node.get("data", {})
    node_id = node.get("id")
    output_variable = node_data.get("outputVariable")
    prompt = (
        node_data.get("researchPrompt")
        or node_data.get("deepResearchPrompt")
        or node_data.get("prompt")
        or ""
    ).strip()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "OPENAI_API_KEY is not set",
        }

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "o3-deep-research",
            "input": prompt,
            "tools": [{"type": "web_search_preview"}],
            "background": True,
        }
        api_response = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=30)
        api_response.raise_for_status()
        response_data = api_response.json()
        thread_id = response_data.get("id") or response_data.get("thread_id")

        if not thread_id:
            return {
                "output": None,
                "outputVariable": output_variable,
                "branch": None,
                "pause": False,
                "error": "OpenAI API did not return a thread id",
            }

        logger.info(f"OpenAI deep research started: thread_id={thread_id}")

        run_ref.update({
            "status": "paused_openai_research",
            "pausedAt": firestore.SERVER_TIMESTAMP,
            "pausedAtNodeId": node_id,
            "openaiResearchJob": {
                "threadId": thread_id,
                "status": "called",
                "nodeId": node_id,
                "outputVariable": output_variable,
            },
            "lockedBy": None,
        })

        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": True,
            "error": None,
        }
    except requests.RequestException as e:
        logger.exception("OpenAI start deep research failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }
    except Exception as e:
        logger.exception("OpenAI start deep research failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }


def _research_perplexity_sonar_pro(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """Perplexity Sonar Pro (sync) — call sonar-pro, return answer + citations like async deep research."""
    node_data = node.get("data", {})
    output_variable = node_data.get("outputVariable")
    prompt = (
        node_data.get("researchPrompt")
        or node_data.get("deepResearchPrompt")
        or node_data.get("prompt")
        or ""
    ).strip()

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "PERPLEXITY_API_KEY is not set",
        }

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
        )
        content = (response.choices[0].message.content or "").strip()
        raw_citations = getattr(response, "citations", None) or []
        citations = [{"title": None, "url": c} if isinstance(c, str) else c for c in raw_citations]
        citation_lines = [f"[{i + 1}] {c.get('url', '')}" for i, c in enumerate(citations)]
        full_response = content
        if citation_lines:
            full_response = content + "\n\n" + "\n".join(citation_lines)

        run_ref.update({
            "citations": [c if isinstance(c, dict) else {"title": None, "url": c} for c in raw_citations],
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })

        return {
            "output": full_response,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }
    except Exception as e:
        logger.exception("Perplexity Sonar Pro failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }


def _research_perplexity(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """Start Perplexity async deep research job, store job id on run, pause for poller to complete."""
    node_data = node.get("data", {})
    node_id = node.get("id")
    output_variable = node_data.get("outputVariable")
    prompt = (
        node_data.get("researchPrompt")
        or node_data.get("deepResearchPrompt")
        or node_data.get("prompt")
        or ""
    ).strip()

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "PERPLEXITY_API_KEY is not set",
        }

    try:
        client = PerplexityAsyncClient(api_key=api_key)
        response = client.create_async_job(prompt)
        thread_id = response.get("id") or response.get("request_id")
        if not thread_id:
            return {
                "output": None,
                "outputVariable": output_variable,
                "branch": None,
                "pause": False,
                "error": "Perplexity API did not return a job id",
            }

        logger.info(f"Perplexity deep research started: thread_id={thread_id}")

        run_ref.update({
            "status": "paused_deep_research",
            "pausedAt": firestore.SERVER_TIMESTAMP,
            "pausedAtNodeId": node_id,
            "deepResearchJob": {
                "provider": "perplexity deep research",
                "threadId": thread_id,
                "status": "processing",
                "nodeId": node_id,
                "outputVariable": output_variable,
            },
            "lockedBy": None,
        })

        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": True,
            "error": None,
        }
    except Exception as e:
        logger.exception("Perplexity create_async_job failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }


RESEARCH_EXECUTORS = {
    "openai": _research_openai,
    "openai deep research": _research_openai,
    "perplexity deep research": _research_perplexity,
    "perplexity sonar pro": _research_perplexity_sonar_pro,
}


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Deep research node router.
    Reads prompt and provider (openai | perplexity deep research | perplexity sonar pro)
    from node config, routes to the matching executor.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Deep Research")

    prompt = (
        node_data.get("researchPrompt")
        or node_data.get("deepResearchPrompt")
        or node_data.get("prompt")
        or ""
    ).strip()
    provider = (
        node_data.get("researchProvider")
        or node_data.get("deepResearchProvider")
        or node_data.get("provider")
        or ""
    ).strip().lower().replace("_", " ")

    if not prompt:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": "Deep research node has no 'prompt' / 'researchPrompt' configured",
        }

    executor = RESEARCH_EXECUTORS.get(provider)
    if not executor:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": f"Unknown research provider: {provider}. Use 'openai', 'perplexity deep research', or 'perplexity sonar pro'.",
        }

    logger.info(f"Deep research node '{label}' using provider: {provider}")
    return executor(node, variables, db, run_ref, run_data)
