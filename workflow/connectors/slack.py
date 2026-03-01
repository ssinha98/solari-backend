import logging
import json
from datetime import datetime, timedelta, timezone
from app import (
    slack_get,
    slack_get_user_name,
    get_bot_token_for_uid,
    slack_ts_to_str,
    is_system_message,
    flatten_slack_transcript,
)

logger = logging.getLogger(__name__)


# ─── Date filter helpers ──────────────────────────────────────────────────────

def get_oldest_ts(slack_filter: str, custom_range: dict) -> str | None:
    """
    Convert slackFilter + slackCustomRange to a Slack timestamp string.
    Returns None for full fetch (no filter).
    """
    now = datetime.now(timezone.utc)

    if slack_filter == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return str(start.timestamp())

    if slack_filter == "last_week":
        start = now - timedelta(days=7)
        return str(start.timestamp())

    if slack_filter == "custom" and custom_range:
        from_date = custom_range.get("fromDate")
        if from_date:
            try:
                dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
                return str(dt.timestamp())
            except Exception:
                logger.warning(f"Could not parse fromDate: {from_date}")
                return None

    return None  # no filter — fetch up to limit


def get_latest_ts(slack_filter: str, custom_range: dict) -> str | None:
    """
    Convert slackCustomRange toDate to a Slack timestamp string.
    Only relevant for custom range — today/last_week use now as latest.
    """
    if slack_filter == "custom" and custom_range:
        to_date = custom_range.get("toDate")
        if to_date:
            try:
                dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
                return str(dt.timestamp())
            except Exception:
                logger.warning(f"Could not parse toDate: {to_date}")
                return None

    return None  # no upper bound


# ─── Message formatting ───────────────────────────────────────────────────────

def format_messages(messages: list, bot_token: str) -> str:
    """
    Format a list of Slack messages into a readable text transcript.
    Filters system messages and resolves user IDs to names.
    """
    lines = []
    user_name_cache: dict[str, str] = {}
    for msg in messages:
        if is_system_message(msg):
            continue

        user_id = msg.get("user", "unknown")
        try:
            user_name = slack_get_user_name(bot_token, user_id, user_name_cache)
        except Exception:
            user_name = user_id

        ts = slack_ts_to_str(msg.get("ts", ""))
        text = msg.get("text", "").strip()

        if text:
            lines.append(f"[{ts}] {user_name}: {text}")

    return "\n".join(lines)


# ─── Main executor ────────────────────────────────────────────────────────────

def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Slack connector executor.

    Fetches messages from a Slack channel using the node config filters,
    formats them into a readable transcript, and returns as output variable.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Slack Connector")

    channel_id = node_data.get("slackChannelId")
    channel_name = node_data.get("slackChannelName", channel_id)
    slack_filter = node_data.get("slackFilter", "today")
    custom_range = node_data.get("slackCustomRange")
    limit = node_data.get("slackMessageCount", 50)
    output_variable = node_data.get("outputVariable") or node_data.get("slackOutputVariable")
    user_id = run_data.get("triggeredBy")

    if not channel_id:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "Slack node has no channel configured",
        }

    logger.info(f"Slack connector '{label}' fetching from #{channel_name} (filter: {slack_filter})")

    try:
        # get bot token for this user
        bot_token, team_id_used = get_bot_token_for_uid(db, user_id)

        # build date filters
        oldest_ts = get_oldest_ts(slack_filter, custom_range)
        latest_ts = get_latest_ts(slack_filter, custom_range)

        # build params
        params = {
            "channel": channel_id,
            "limit": limit,
        }
        if oldest_ts:
            params["oldest"] = oldest_ts
        if latest_ts:
            params["latest"] = latest_ts

        # fetch messages
        data = slack_get(bot_token, "conversations.history", params)
        messages = data.get("messages", [])

        logger.info(f"Slack connector '{label}' fetched {len(messages)} messages")

        # format into readable transcript
        transcript = format_messages(messages, bot_token)

        if not transcript:
            transcript = f"No messages found in #{channel_name} for the selected time range."

        return {
            "output": transcript,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except RuntimeError as e:
        slack_error = e.args[0]
        error_msg = f"Slack API error: {slack_error.get('error', 'unknown')}"
        logger.error(f"Slack connector '{label}' failed: {error_msg}")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": error_msg,
        }

    except Exception as e:
        logger.error(f"Slack connector '{label}' failed: {e}", exc_info=True)
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }