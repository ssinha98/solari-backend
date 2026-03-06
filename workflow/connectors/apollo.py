import logging
from workflow.nodes import apollo as apollo_node

logger = logging.getLogger(__name__)


def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Apollo connector: delegates to the Apollo enrich node.
    Supports connector field names: apolloPersonFullName, apolloPersonCompany.
    """
    return apollo_node.execute(node, variables, db, run_ref, run_data)
