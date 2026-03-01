import logging
from app import keywordsai_chat_completion

logger = logging.getLogger(__name__)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    LLM node executor.

    Reads model, systemPrompt, userPrompt and temperature from node config,
    calls the model via KeywordsAI, and returns the text output.

    By the time this is called, interpolator has already replaced any
    {{variable_name}} references in the prompts with real values.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "LLM")

    model = node_data.get("model", "gpt-4o")
    system_prompt = node_data.get("systemPrompt", "")
    user_prompt = node_data.get("userPrompt", "")
    temperature = node_data.get("temperature", 0.7)
    output_variable = node_data.get("outputVariable")
    user_id = run_data.get("triggeredBy")

    if not user_prompt:
        raise Exception(f"LLM node '{label}' has no user prompt configured")

    # build messages
    messages = []
    if system_prompt:
        messages.append({ "role": "system", "content": system_prompt })
    messages.append({ "role": "user", "content": user_prompt })

    logger.info(f"LLM node '{label}' calling model: {model}")

    try:
        response = keywordsai_chat_completion(
            messages=messages,
            model=model,
            user_id=user_id,
            temperature=temperature,
        )

        output = response["choices"][0]["message"]["content"]
        logger.info(f"LLM node '{label}' completed, output length: {len(output)} chars")

        return {
            "output": output,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except Exception as e:
        logger.error(f"LLM node '{label}' failed: {e}", exc_info=True)
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }