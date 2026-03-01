import logging
import json
from app import _get_issue

logger = logging.getLogger(__name__)


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_ticket(ticket: dict) -> str:
    """Format a single Jira ticket into readable text."""
    fields = ticket.get("fields", {})

    key = ticket.get("key", "Unknown")
    summary = fields.get("summary", "No summary")
    status = fields.get("status", {}).get("name", "Unknown")
    issue_type = fields.get("issuetype", {}).get("name", "Unknown")
    priority = fields.get("priority", {}).get("name", "Unknown")
    assignee = fields.get("assignee") or {}
    assignee_name = assignee.get("displayName", "Unassigned")
    project = fields.get("project", {}).get("name", "Unknown")
    created = fields.get("created", "")[:10]  # just the date portion

    # description
    description = fields.get("description", "")
    if isinstance(description, dict):
        # Jira returns description as Atlassian Document Format (ADF) — extract plain text
        description = extract_adf_text(description)
    description = (description or "No description").strip()[:500]  # cap at 500 chars

    # comments
    comments = fields.get("comment", {}).get("comments", [])
    comment_lines = []
    for c in comments[-3:]:  # last 3 comments only
        author = c.get("author", {}).get("displayName", "Unknown")
        body = c.get("body", "")
        if isinstance(body, dict):
            body = extract_adf_text(body)
        comment_lines.append(f"  [{author}]: {body[:200]}")
    comments_text = "\n".join(comment_lines) if comment_lines else "  No comments"

    return (
        f"Ticket: {key}\n"
        f"Project: {project}\n"
        f"Type: {issue_type} | Status: {status} | Priority: {priority}\n"
        f"Assignee: {assignee_name} | Created: {created}\n"
        f"Summary: {summary}\n"
        f"Description: {description}\n"
        f"Recent comments:\n{comments_text}"
    )


def extract_adf_text(adf: dict) -> str:
    """
    Recursively extract plain text from Atlassian Document Format (ADF).
    Jira returns description/comments as ADF dicts rather than plain strings.
    """
    if not adf or not isinstance(adf, dict):
        return ""

    text_parts = []
    if adf.get("type") == "text":
        text_parts.append(adf.get("text", ""))

    for child in adf.get("content", []):
        text_parts.append(extract_adf_text(child))

    return " ".join(filter(None, text_parts))


def format_tickets(tickets: list) -> str:
    """Format a list of Jira tickets into a readable transcript."""
    if not tickets:
        return "No tickets found."

    sections = []
    for i, ticket in enumerate(tickets, 1):
        sections.append(f"--- Ticket {i} ---\n{format_ticket(ticket)}")

    return "\n\n".join(sections)


# ─── Main executor ────────────────────────────────────────────────────────────

def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Jira connector executor.

    Fetches full ticket details for each pre-selected ticket in the node config,
    formats them into readable text, and returns as output variable.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Jira Connector")

    selected_tickets = node_data.get("jiraSelectedTickets", [])
    output_variable = node_data.get("outputVariable")
    user_id = run_data.get("triggeredBy")

    if not selected_tickets:
        return {
            "output": "No Jira tickets configured for this node.",
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    logger.info(f"Jira connector '{label}' fetching {len(selected_tickets)} tickets")

    try:
        tickets = []
        for selected in selected_tickets:
            ticket_id = selected.get("id") or selected.get("key")
            if not ticket_id:
                continue

            logger.info(f"Fetching Jira ticket: {ticket_id}")
            ticket = _get_issue(user_id, ticket_id)

            tickets.append(ticket)

        logger.info(f"Jira connector '{label}' fetched {len(tickets)} tickets successfully")

        transcript = format_tickets(tickets)

        return {
            "output": transcript,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Jira connector '{label}' failed: {e}", exc_info=True)
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }