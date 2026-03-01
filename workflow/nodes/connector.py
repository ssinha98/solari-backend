import logging
from workflow.connectors import slack, jira, confluence

logger = logging.getLogger(__name__)

CONNECTOR_EXECUTORS = {
    "slack":       slack.execute,
    "jira":        jira.execute,
    "confluence":  confluence.execute,
}

def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Connector node router.
    Reads connectorService from node data and routes to the correct connector.
    """
    node_data = node.get("data", {})
    service = node_data.get("connectorService")

    if not service:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": "Connector node has no connectorService configured",
        }

    executor = CONNECTOR_EXECUTORS.get(service)
    if not executor:
        return {
            "output": None,
            "outputVariable": node_data.get("outputVariable"),
            "branch": None,
            "pause": False,
            "error": f"Unknown connector service: {service}",
        }

    logger.info(f"Routing to connector: {service}")
    return executor(node, variables, db, run_ref, run_data)