import logging
from app import send_email

logger = logging.getLogger(__name__)

DEFAULT_FROM = "postmaster@robots.usesolari.ai"


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Email node executor.

    Sends an email. Reads from, to, subject, body from node config
    (after interpolation). If 'from' is not set, defaults to
    postmaster@robots.usesolari.ai.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Email")

    # Support frontend field names (emailFrom, emailTo, etc.) and shorthand (from, to, etc.)
    from_addr = node_data.get("emailFrom") or node_data.get("from") or DEFAULT_FROM
    to_addr = node_data.get("emailTo") or node_data.get("to")
    subject = node_data.get("emailSubject") or node_data.get("subject", "")
    body = node_data.get("emailBody") or node_data.get("body", "")

    if not to_addr:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": "Email node has no 'to' / 'emailTo' address configured",
        }

    logger.info(f"Email node '{label}' sending to {to_addr}")

    try:
        response = send_email(
            to_email=to_addr,
            subject=subject,
            body=body,
            from_email=from_addr,
        )
        if response is None or not (200 <= getattr(response, "status_code", 0) < 300):
            return {
                "output": None,
                "outputVariable": node_data.get("outputVariable"),
                "branch": None,
                "pause": False,
                "error": "Failed to send email",
            }
        return {
            "output": "sent",
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": None,
        }
    except Exception as e:
        logger.exception(f"Email node '{label}' failed")
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": str(e),
        }
