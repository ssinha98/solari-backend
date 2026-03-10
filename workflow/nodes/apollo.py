import logging
import os
import json
import requests
from app import keywordsai_chat_completion
import time
import random

logger = logging.getLogger(__name__)

APOLLO_MATCH_URL = "https://api.apollo.io/api/v1/people/match"


def _sleep_with_jitter(base_seconds: float = 0.3):
    """Proactive delay with jitter to avoid hitting rate limits."""
    time.sleep(base_seconds + random.uniform(0, 0.5))


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    if isinstance(e, requests.HTTPError):
        return e.response is not None and e.response.status_code == 429
    return "rate limit" in str(e).lower()


def _apollo_request_with_retry(url: str, headers: dict, params: dict, retries: int = 3) -> dict:
    """
    Make an Apollo API request with exponential backoff + jitter on rate limit errors.
    Applies a proactive jitter delay after every successful call.
    """
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            _sleep_with_jitter()  # proactive delay after every successful call
            return response.json()
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Apollo rate limited. Retrying in {delay:.2f}s (attempt {attempt + 1}/{retries})...")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError("Apollo request failed after all retries")


def _get_apollo_api_key(db, team_id: str) -> str | None:
    """
    Get Apollo API key: first from team doc (APOLLO_API_KEY), then from env.
    """
    if not team_id:
        return os.getenv("APOLLO_API_KEY")

    try:
        team_snap = db.collection("teams").document(team_id).get()
        if team_snap.exists:
            team_data = team_snap.to_dict() or {}
            key = team_data.get("APOLLO_API_KEY") or team_data.get("apollo_api_key")
            if key:
                return key
    except Exception as e:
        logger.warning(f"Could not fetch team doc for Apollo key: {e}")

    return os.getenv("APOLLO_API_KEY")


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Apollo Enrich node executor.

    Enriches a lead using Apollo.io People Match API.
    Reads personName (or person_name/name) and company (or organization_name)
    from node config (after interpolation). Optionally runs OpenAI analysis
    over the Apollo data if a prompt is configured.

    Output:
    - If prompt is set: only the analysis string.
    - If no prompt: the full Apollo API response.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Apollo Enrich")

    # Support frontend field names: personName, person_name, name, apolloPersonFullName
    name = (
        node_data.get("personName")
        or node_data.get("person_name")
        or node_data.get("name")
        or node_data.get("apolloPersonFullName")
        or ""
    ).strip()

    # Support frontend field names: company, organization_name, apolloPersonCompany
    company = (
        node_data.get("company")
        or node_data.get("organization_name")
        or node_data.get("apolloPersonCompany")
        or ""
    ).strip()

    prompt = (
        node_data.get("prompt")
        or node_data.get("analysisPrompt")
        or node_data.get("apolloPrompt")
        or node_data.get("apolloPersonPrompt")
        or ""
    ).strip()
    output_variable = node_data.get("outputVariable")
    team_id = run_data.get("teamId")
    user_id = run_data.get("triggeredBy")

    if not name:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "Apollo node has no 'personName' / 'name' configured",
        }

    if not company:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "Apollo node has no 'company' / 'organization_name' configured",
        }

    api_key = _get_apollo_api_key(db, team_id)
    if not api_key:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "APOLLO_API_KEY not set on team doc or in environment",
        }

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    params = {
        "name": name,
        "organization_name": company,
        "reveal_personal_emails": False,
        "reveal_phone_number": False,
    }

    logger.info(f"Apollo node '{label}' enriching: name={name!r}, company={company!r}")

    try:
        apollo_result = _apollo_request_with_retry(APOLLO_MATCH_URL, headers, params)

        logger.info(
            "Apollo node full response: %s",
            json.dumps(apollo_result, indent=2, default=str),
        )

        person = apollo_result.get("person")
        if not person:
            return {
                "output": None,
                "outputVariable": output_variable,
                "branch": None,
                "pause": False,
                "error": f"Apollo found no match for {name} at {company}",
            }

        # If no prompt, return raw Apollo data
        if not prompt:
            return {
                "output": apollo_result,
                "outputVariable": output_variable,
                "branch": None,
                "pause": False,
                "error": None,
            }

        # If prompt, run KeywordsAI over Apollo data and return only analysis
        apollo_summary = json.dumps(apollo_result, indent=2, default=str)
        messages = [
            {
                "role": "user",
                "content": f"Apollo enrichment data:\n\n{apollo_summary}\n\n{prompt}",
            },
        ]

        openai_response = keywordsai_chat_completion(
            messages=messages,
            model=node_data.get("model", "gpt-4o-mini"),
            user_id=user_id,
            temperature=node_data.get("temperature", 0.3),
        )

        analysis = openai_response["choices"][0]["message"]["content"]

        return {
            "output": analysis,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except requests.RequestException as e:
        logger.exception(f"Apollo node '{label}' API request failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }
    except Exception as e:
        logger.exception(f"Apollo node '{label}' failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }