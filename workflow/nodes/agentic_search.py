import json
import logging
from app import firecrawl
from perplexity import Perplexity

logger = logging.getLogger(__name__)


def _to_json_serializable(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return {"raw": str(obj)}


def _strip_domain_url(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    return raw.strip().removeprefix("https://").removeprefix("http://").strip("/")


def _normalize_sources(sources) -> list | None:
    if sources is None:
        return None
    if isinstance(sources, list):
        return [s for s in sources if s] if sources else None
    if isinstance(sources, str) and sources.strip():
        return [sources.strip()]
    return None


def _normalize_domain_filter(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [_strip_domain_url(s) for s in raw if _strip_domain_url(s)]
    if isinstance(raw, str) and raw.strip():
        parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
        return [_strip_domain_url(p) for p in parts if _strip_domain_url(p)]
    return []


def _format_as_string(results: list, source_type: str) -> str:
    """Format a list of results as a readable string for single mode."""
    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        description = r.get("description") or r.get("snippet", "")
        date = r.get("date", "")

        line = f"- {title} ({url})"
        if description:
            line += f"\n  {description}"
        if date:
            line += f"\n  Date: {date}"
        lines.append(line)
    return "\n\n".join(lines)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    node_data = node.get("data", {})
    label = node_data.get("label", "Agentic Search")
    output_variable = node_data.get("outputVariable")
    output_type = run_data.get("outputType", "single")
    engine = node_data.get("engine") or node_data.get("agenticSearchEngine") or "firecrawl"

    query = (node_data.get("agenticSearchQuery") or "").strip()
    sources = _normalize_sources(node_data.get("agenticSearchSources"))
    site = _strip_domain_url(node_data.get("agenticSearchWebsite") or "")
    limit = node_data.get("agenticSearchLimit", 10)
    if limit is not None and not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10

    if not query:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "Agentic search node has no 'agenticSearchQuery' configured",
        }

    if engine == "perplexity":
        domain_filter = _normalize_domain_filter(
            node_data.get("agenticSearchDomainFilter") or node_data.get("agenticSearchWebsite")
        )
        return _execute_perplexity(
            query=query,
            domain_filter=domain_filter,
            limit=limit,
            output_variable=output_variable,
            output_type=output_type,
            label=label,
        )

    # Firecrawl path
    if firecrawl is None:
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": "FIRECRAWL_API_KEY is not set",
        }

    if site:
        query = f"site:{site} {query}"

    search_params = {"query": query, "limit": limit}
    if sources:
        search_params["sources"] = sources

    logger.info(f"Agentic search node '{label}' query={query!r}, sources={sources}, limit={limit}, output_type={output_type}")

    try:
        result = firecrawl.search(**search_params)
        output = _to_json_serializable(result)

        # extract the right list based on source type
        web_results = output.get("web") or []
        news_results = output.get("news") or []

        # prefer whichever has results, news takes priority if sources includes news
        if sources and "news" in sources and news_results:
            results = news_results
            source_type = "news"
        elif web_results:
            results = web_results
            source_type = "web"
        elif news_results:
            results = news_results
            source_type = "news"
        else:
            results = []
            source_type = "web"

        if output_type == "table":
            final_output = results  # list for graph.py to detect as row producer
        else:
            final_output = _format_as_string(results, source_type)

        return {
            "output": final_output,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Agentic search node '{label}' failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }


def _execute_perplexity(
    query: str,
    domain_filter: list[str],
    limit: int,
    output_variable: str | None,
    output_type: str,
    label: str,
) -> dict:
    try:
        client = Perplexity()
        search_params = {"query": query, "max_results": limit}
        if domain_filter:
            search_params["search_domain_filter"] = domain_filter

        result = client.search.create(**search_params)
        output = _to_json_serializable(result)

        results = output.get("results") or []

        if output_type == "table":
            final_output = results  # list for graph.py to detect as row producer
        else:
            final_output = _format_as_string(results, "perplexity")

        return {
            "output": final_output,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Agentic search node '{label}' (Perplexity) failed")
        return {
            "output": None,
            "outputVariable": output_variable,
            "branch": None,
            "pause": False,
            "error": str(e),
        }