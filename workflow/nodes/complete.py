import logging
from firebase_admin import firestore

logger = logging.getLogger(__name__)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Done node executor.
    Simply marks the run as completed. Any actions (Slack, S3 etc.)
    should be handled by connector/action nodes before this node.
    """
    label = node.get("data", {}).get("label", "Done")
    logger.info(f"Done node '{label}' — marking run as completed")

    run_ref.update({
        "status": "completed",
        "completedAt": firestore.SERVER_TIMESTAMP,
        "lockedBy": None,
    })

    return {
        "output": "completed",
        "outputVariable": None,
        "branch": None,
        "pause": False,
        "error": None,
    }