import logging
from datetime import datetime, timezone
from firebase_config import firestore
from workflow.interpolator import interpolate
from workflow.nodes import llm, connector, eval_node, trigger, for_review, complete
# from workflow.nodes import llm, connector, eval_node, trigger, output, for_review, complete
#
logger = logging.getLogger(__name__)

# ─── Node executor routing ────────────────────────────────────────────────────

NODE_EXECUTORS = {
    "trigger":   trigger.execute,
    "llm":       llm.execute,
    "connector": connector.execute,
    "eval":      eval_node.execute,
    "forReview": for_review.execute,
    "complete":   complete.execute,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_lookup_maps(nodes, edges):
    """Build id-keyed maps for fast lookup during traversal."""
    nodes_by_id = { n["id"]: n for n in nodes }
    edges_by_source = {}
    for edge in edges:
        edges_by_source.setdefault(edge["source"], []).append(edge)
    return nodes_by_id, edges_by_source


def find_trigger_node(nodes, edges):
    """Find the start node — the one with no incoming edges."""
    target_ids = { e["target"] for e in edges }
    trigger_nodes = [n for n in nodes if n["id"] not in target_ids]
    if not trigger_nodes:
        raise Exception("No trigger node found — workflow has no entry point")
    if len(trigger_nodes) > 1:
        raise Exception(f"Multiple trigger nodes found: {[n['id'] for n in trigger_nodes]}")
    return trigger_nodes[0]


def pick_next_node(current_node, outgoing_edges, result, nodes_by_id):
    """
    Given the current node and its result, pick the next node to execute.
    Handles branching for eval and condition nodes.
    """
    kind = current_node["data"]["kind"]

    if not outgoing_edges:
        return None

    # eval and condition nodes branch based on result
    if kind == "eval":
        branch = result.get("branch")  # "pass" or "fail"
        if not branch:
            raise Exception(f"Eval node {current_node['id']} returned no branch")
        for edge in outgoing_edges:
            edge_data = edge.get("data") or {}
            if edge_data.get("evalEdgeType") == branch:
                return nodes_by_id[edge["target"]]
        raise Exception(f"No edge found for eval branch: {branch}")

    if kind == "condition":
        branch = str(result.get("branch", "")).lower()  # "true" or "false"
        if not branch:
            raise Exception(f"Condition node {current_node['id']} returned no branch")
        for edge in outgoing_edges:
            edge_data = edge.get("data") or {}
            if edge_data.get("condition") == branch:
                return nodes_by_id[edge["target"]]
        raise Exception(f"No edge found for condition branch: {branch}")

    # all other nodes — single outgoing edge
    return nodes_by_id[outgoing_edges[0]["target"]]


def write_node_execution(run_ref, node, result):
    """Write a node execution record to the run doc."""
    execution = {
        "nodeId":        node["id"],
        "nodeLabel":     node["data"].get("label"),
        "kind":          node["data"].get("kind"),
        "status":        "completed" if not result.get("error") else "failed",
        "output":        result.get("output"),
        "outputVariable": result.get("outputVariable"),
        "error":         result.get("error"),
        "completedAt":   datetime.now(timezone.utc),
    }
    run_ref.update({
        "nodeExecutions": firestore.ArrayUnion([execution]),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })

# ─── Main executor ────────────────────────────────────────────────────────────

def execute_workflow(db, run_ref, run_data):
    """
    Main entry point. Fetches the workflow config, walks the graph,
    executes each node in sequence, and writes results to Firebase.
    """
    agent_ref = run_ref.parent.parent
    version_id = run_data.get("versionId")

    # fetch workflow config
    version_snap = (
        agent_ref.collection("versions")
                 .document(version_id)
                 .get()
    )
    if not version_snap.exists:
        raise Exception(f"Version {version_id} not found")

    version = version_snap.to_dict()
    nodes = version.get("nodes", [])
    edges = version.get("edges", [])

    if not nodes:
        raise Exception("Workflow has no nodes")

    # build lookup maps
    nodes_by_id, edges_by_source = build_lookup_maps(nodes, edges)

    # find start node
    current_node = find_trigger_node(nodes, edges)

    # runtime variable map — starts with input variables
    variables = run_data.get("variables", {}).copy()

    # cycle protection
    visited = set()

    logger.info(f"Starting workflow execution from trigger node: {current_node['id']}")

    while current_node:
        node_id = current_node["id"]
        node_kind = current_node["data"].get("kind")

        # cycle detection
        if node_id in visited:
            raise Exception(f"Cycle detected at node {node_id}")
        visited.add(node_id)

        logger.info(f"Executing node: {node_id} ({node_kind})")

        # get executor for this node type
        executor_fn = NODE_EXECUTORS.get(node_kind)
        if not executor_fn:
            raise Exception(f"Unknown node kind: {node_kind}")

        # interpolate variables into node config before executing
        interpolated_node = interpolate(current_node, variables)

        # execute the node
        result = executor_fn(
            node=interpolated_node,
            variables=variables,
            db=db,
            run_ref=run_ref,
            run_data=run_data,
        )

        # write node execution to Firebase
        write_node_execution(run_ref, current_node, result)

        # handle pause (forReview node)
        if result.get("pause"):
            logger.info(f"Run paused at node {node_id} for review")
            return

        # handle node failure
        if result.get("error"):
            raise Exception(f"Node {node_id} failed: {result['error']}")

        # write output variable to run doc if set
        output_variable = result.get("outputVariable")
        output_value = result.get("output")
        if output_variable and output_value is not None:
            variables[output_variable] = output_value
            run_ref.update({
                f"variables.{output_variable}": output_value,
                f"outputVariables.{output_variable}": output_value,
            })

        # find next node
        outgoing = edges_by_source.get(node_id, [])
        if not outgoing:
            # no more edges — workflow complete
            logger.info(f"Workflow completed — no more edges from node {node_id}")
            run_ref.update({
                "status": "completed",
                "completedAt": firestore.SERVER_TIMESTAMP,
                "lockedBy": None,
            })
            return

        current_node = pick_next_node(current_node, outgoing, result, nodes_by_id)

    # reached end of graph
    run_ref.update({
        "status": "completed",
        "completedAt": firestore.SERVER_TIMESTAMP,
        "lockedBy": None,
    })
    logger.info("Workflow execution complete")