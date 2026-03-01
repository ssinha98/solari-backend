import logging
import requests
from app import _get_jira_creds, confluence_storage_html_to_rag_text

logger = logging.getLogger(__name__)


# ─── API helpers ──────────────────────────────────────────────────────────────

def confluence_get(creds: dict, path: str, params: dict = None) -> dict:
    """
    Make an authenticated GET request to the Confluence API.
    Raises RuntimeError on non-200 responses.
    """
    cloud_id = creds["cloud_id"]
    access_token = creds["access_token"]
    base_url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki"

    response = requests.get(
        f"{base_url}{path}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        params=params or {},
    )

    if response.status_code >= 400:
        raise RuntimeError({
            "status_code": response.status_code,
            "error": response.text,
        })

    return response.json()


def fetch_page(creds: dict, page_id: str) -> str:
    """Fetch a single Confluence page and return as clean plaintext."""
    page = confluence_get(
        creds,
        f"/api/v2/pages/{page_id}",
        params={"body-format": "storage"},
    )
    title = page.get("title", "Untitled")
    storage_html = page.get("body", {}).get("storage", {}).get("value", "")
    text = confluence_storage_html_to_rag_text(storage_html)
    return f"# {title}\n\n{text}"


def search_pages(creds: dict, cql: str, limit: int = 10) -> list[str]:
    """Search Confluence pages by CQL and return list of plaintext page contents."""
    data = confluence_get(
        creds,
        "/rest/api/search",
        params={"cql": cql, "limit": limit},
    )
    results = data.get("results", [])
    pages = []
    for result in results:
        page_id = result.get("content", {}).get("id")
        if page_id:
            try:
                pages.append(fetch_page(creds, page_id))
            except Exception as e:
                logger.warning(f"Failed to fetch page {page_id}: {e}")
    return pages


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_pages(pages: list[str]) -> str:
    """Join multiple page contents into a single transcript."""
    if not pages:
        return "No Confluence pages found."
    return "\n\n---\n\n".join(pages)


# ─── Main executor ────────────────────────────────────────────────────────────

def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Confluence connector executor.

    Supports two modes:
    - confluenceSelectedPages — fetch specific pre-selected pages by ID
    - confluenceCql — search pages using a CQL query

    Returns formatted plaintext of all fetched pages as output variable.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Confluence Connector")

    selected_pages = node_data.get("confluenceSelectedPages", [])
    cql = node_data.get("confluenceCql")
    cql_limit = node_data.get("confluenceCqlLimit", 10)
    output_variable = node_data.get("outputVariable")
    user_id = run_data.get("triggeredBy")

    if not selected_pages and not cql:
        return {
            "output": "No Confluence pages or CQL query configured for this node.",
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    logger.info(f"Confluence connector '{label}' starting fetch")

    try:
        creds = _get_jira_creds(user_id)
        pages = []

        # mode 1 — fetch specific selected pages
        if selected_pages:
            logger.info(f"Fetching {len(selected_pages)} selected Confluence pages")
            for page in selected_pages:
                page_id = page.get("id")
                if not page_id:
                    continue
                try:
                    pages.append(fetch_page(creds, page_id))
                except Exception as e:
                    logger.warning(f"Failed to fetch Confluence page {page_id}: {e}")

        # mode 2 — CQL search
        elif cql:
            logger.info(f"Searching Confluence with CQL: {cql}")
            pages = search_pages(creds, cql, limit=cql_limit)

        logger.info(f"Confluence connector '{label}' fetched {len(pages)} pages")

        transcript = format_pages(pages)

        return {
            "output": transcript,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except RuntimeError as e:
        error_detail = e.args[0]
        error_msg = f"Confluence API error: {error_detail.get('status_code')} — {error_detail.get('error', '')[:200]}"
        logger.error(f"Confluence connector '{label}' failed: {error_msg}")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": error_msg,
        }

    except Exception as e:
        logger.error(f"Confluence connector '{label}' failed: {e}", exc_info=True)
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }