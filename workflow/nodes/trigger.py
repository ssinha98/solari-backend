import logging

logger = logging.getLogger(__name__)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Trigger node executor.
    
    The trigger node is the entry point of the workflow. It doesn't do any
    processing — it just confirms the run has started and passes the input
    variables downstream so subsequent nodes can reference them.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Trigger")

    logger.info(f"Trigger node '{label}' fired")

    # input variables are already in the variables map from create_run
    # just return them as the output so they're available downstream
    return {
        "output": variables.copy(),
        "outputVariable": None,  # trigger doesn't save to a named variable
        "branch": None,
        "pause": False,
        "error": None,
    }