import logging
import json
from app import keywordsai_chat_completion

logger = logging.getLogger(__name__)

# ─── Default judge prompt ─────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are a strict but fair evaluator. Your job is to assess whether a given text meets the specified criteria.

You must respond with valid JSON only — no preamble, no explanation outside the JSON.

Response format:
{
  "pass": true or false,
  "score": 0.0 to 1.0,
  "reason": "brief explanation of your decision"
}"""

# ─── Main judge function ──────────────────────────────────────────────────────

def evaluate(
    input_text: str,
    criteria: str,
    model: str,
    system_prompt: str,
    threshold: float,
    user_id: str,
) -> dict:
    """
    Run an LLM judge evaluation on input_text against the given criteria.

    Returns:
    {
        "pass": bool,
        "score": float,
        "reason": str,
        "branch": "pass" | "fail"
    }
    """
    # use custom system prompt if provided, otherwise use default
    effective_system_prompt = system_prompt.strip() if system_prompt else DEFAULT_SYSTEM_PROMPT

    user_prompt = f"""Criteria: {criteria}

Text to evaluate:
{input_text}

Does the text meet the criteria? Respond with JSON only."""

    messages = [
        { "role": "system", "content": effective_system_prompt },
        { "role": "user",   "content": user_prompt },
    ]

    logger.info(f"LLM judge calling model: {model} with threshold: {threshold}")

    response = keywordsai_chat_completion(
        messages=messages,
        model=model,
        user_id=user_id,
        temperature=0.0,  # deterministic for evals
    )

    raw = response["choices"][0]["message"]["content"]

    # strip markdown code fences if model wraps in ```json
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        raise Exception(f"LLM judge returned invalid JSON: {raw}") from e

    score = float(result.get("score", 0.0))
    passed = result.get("pass", score >= threshold)

    # override with threshold check if score is present
    if "score" in result:
        passed = score >= threshold

    branch = "pass" if passed else "fail"

    logger.info(f"LLM judge result: {branch} (score: {score}, threshold: {threshold})")

    return {
        "pass": passed,
        "score": score,
        "reason": result.get("reason", ""),
        "branch": branch,
    }