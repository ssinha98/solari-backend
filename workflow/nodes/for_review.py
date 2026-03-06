import logging
from firebase_config import firestore

logger = logging.getLogger(__name__)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    ForReview node executor.

    Creates a forReview document in Firestore and pauses the run.
    Returns pause: True to tell graph.py to stop execution.

    The input to review is the output of the preceding node —
    taken from the last written variable in the variables map.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Human Review")
    notes = node_data.get("notes", "")
    node_id = node.get("id")

    agent_id   = run_data.get("agentId")
    agent_name = run_data.get("agentName")
    team_id    = run_data.get("teamId")
    run_id     = run_ref.id
    output_type = run_data.get("outputType", "single")

    input_text = str(list(variables.values())[-1]) if variables else ""

    logger.info(f"ForReview node '{label}' creating review task")

    try:
        review_ref = (
            db.collection("teams").document(team_id)
              .collection("forReview").document()
        )
        review_ref.set({
            "teamId":     team_id,
            "agentId":    agent_id,
            "agentName":  agent_name,
            "runId":      run_id,
            "nodeId":     node_id,
            "nodeLabel":  label,
            "input":      input_text,
            "output":     "",
            "notes":      notes,
            "outputType": output_type,
            "status":     "pending",
            "createdAt":  firestore.SERVER_TIMESTAMP,
            "resolvedAt": None,
            "resolvedBy": None,
        })

        review_id = review_ref.id
        logger.info(f"ForReview doc created: {review_id}")

        # pause the run
        run_ref.update({
            "status": "paused",
            "pausedAt": firestore.SERVER_TIMESTAMP,
            "pausedAtNodeId": node_id,
            "blockedOn": {
                "reason": "forReview",
                "reviewId": review_id,
            },
            "lockedBy": None,
        })

        logger.info(f"Run {run_id} paused at node {node_id}")

        return {
            "output": f"Flagged for review: {label}",
            "outputVariable": None,
            "branch": None,
            "pause": True,          # tells graph.py to stop execution
            "error": None,
            "reviewId": review_id,
        }

    except Exception as e:
        logger.error(f"ForReview node '{label}' failed: {e}", exc_info=True)
        return {
            "output": None,
            "outputVariable": None,
            "branch": None,
            "pause": True,          # still pause even on error — don't continue past a review node
            "error": str(e),
        }