import logging
import re
from workflow.evals import llm_judge

logger = logging.getLogger(__name__)


# ─── Regex eval ───────────────────────────────────────────────────────────────

def evaluate_regex(input_text: str, pattern: str) -> dict:
    """
    Check if input_text matches the given regex pattern.
    Passes if a match is found.
    """
    try:
        match = re.search(pattern, input_text)
        passed = match is not None
        return {
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "reason": f"Pattern '{pattern}' {'matched' if passed else 'did not match'}",
            "branch": "pass" if passed else "fail",
        }
    except re.error as e:
        raise Exception(f"Invalid regex pattern '{pattern}': {e}")


# ─── Main executor ────────────────────────────────────────────────────────────

def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Eval node executor.

    Reads evalType from node data and routes to the correct eval implementation.
    Always returns a branch of "pass" or "fail" for graph traversal.

    The input to evaluate is the output of the preceding node — the graph
    passes it in via variables under the key of the preceding node's outputVariable.
    Since eval nodes don't have an explicit inputVariable, we use the most
    recently written variable as the input.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Eval")
    eval_type = node_data.get("evalType", "llm_judge")
    user_id = run_data.get("triggeredBy")

    # get input — eval nodes evaluate the output of the preceding node
    # evalInputVariable can be explicitly set, otherwise use last variable written
    eval_input_variable = node_data.get("evalInputVariable")
    if eval_input_variable and eval_input_variable in variables:
        input_text = str(variables[eval_input_variable])
    else:
        # fall back to last written variable value
        input_text = str(list(variables.values())[-1]) if variables else ""

    if not input_text:
        return {
            "output": "fail — no input to evaluate",
            "outputVariable": None,
            "branch": "fail",
            "pause": False,
            "error": "No input text found to evaluate",
        }

    logger.info(f"Eval node '{label}' running {eval_type} eval")

    try:
        if eval_type == "llm_judge":
            result = llm_judge.evaluate(
                input_text=input_text,
                criteria=node_data.get("evalCriteria", ""),
                model=node_data.get("evalModel", "gpt-4o"),
                system_prompt=node_data.get("evalSystemPrompt", ""),
                threshold=float(node_data.get("evalThreshold", 0.7)),
                user_id=user_id,
            )

        elif eval_type == "regex":
            result = evaluate_regex(
                input_text=input_text,
                pattern=node_data.get("evalCriteria", ""),
            )

        else:
            raise Exception(f"Unsupported eval type: {eval_type}")

        return {
            "output": result["branch"],  # "pass" or "fail"
            "outputVariable": None,      # eval doesn't save to a variable
            "branch": result["branch"],
            "pause": False,
            "error": None,
            # store eval details for node execution record
            "evalResult": {
                "pass": result["pass"],
                "score": result.get("score"),
                "reason": result.get("reason"),
                "evalType": eval_type,
            },
        }

    except Exception as e:
        logger.error(f"Eval node '{label}' failed: {e}", exc_info=True)
        return {
            "output": "fail",
            "outputVariable": None,
            "branch": "fail",  # fail safe on error
            "pause": False,
            "error": str(e),
        }