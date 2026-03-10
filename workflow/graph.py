import re
import time
import random
import logging
from datetime import datetime, timezone
from firebase_admin import firestore
from workflow.interpolator import interpolate
from workflow.nodes import (
    llm,
    connector,
    eval_node,
    trigger,
    complete,
    for_review,
    pipedrive,
    email,
    website,
    deep_research,
    apollo,
    agentic_search,
)

logger = logging.getLogger(__name__)

# ─── Node executor routing ────────────────────────────────────────────────────

NODE_EXECUTORS = {
    "trigger":       trigger.execute,
    "llm":           llm.execute,
    "connector":     connector.execute,
    "eval":          eval_node.execute,
    "complete":      complete.execute,
    "forReview":     for_review.execute,
    "pipedrive":     pipedrive.execute,
    "email":         email.execute,
    "website":       website.execute,
    "deepResearch":  deep_research.execute,
    "apolloEnrich":  apollo.execute,
    "agenticSearch": agentic_search.execute,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_lookup_maps(nodes, edges):
    nodes_by_id = { n["id"]: n for n in nodes }
    edges_by_source = {}
    for edge in edges:
        edges_by_source.setdefault(edge["source"], []).append(edge)
    return nodes_by_id, edges_by_source


def find_trigger_node(nodes, edges):
    target_ids = { e["target"] for e in edges }
    trigger_nodes = [n for n in nodes if n["id"] not in target_ids]
    if not trigger_nodes:
        raise Exception("No trigger node found — workflow has no entry point")
    if len(trigger_nodes) > 1:
        raise Exception(f"Multiple trigger nodes found: {[n['id'] for n in trigger_nodes]}")
    return trigger_nodes[0]


def pick_next_node(current_node, outgoing_edges, result, nodes_by_id):
    kind = current_node["data"]["kind"]

    if not outgoing_edges:
        return None

    if kind == "eval":
        branch = result.get("branch")
        if not branch:
            raise Exception(f"Eval node {current_node['id']} returned no branch")
        for edge in outgoing_edges:
            edge_data = edge.get("data") or {}
            if edge_data.get("evalEdgeType") == branch:
                return nodes_by_id[edge["target"]]
        raise Exception(f"No edge found for eval branch: {branch}")

    if kind == "condition":
        branch = str(result.get("branch", "")).lower()
        if not branch:
            raise Exception(f"Condition node {current_node['id']} returned no branch")
        for edge in outgoing_edges:
            edge_data = edge.get("data") or {}
            if edge_data.get("condition") == branch:
                return nodes_by_id[edge["target"]]
        raise Exception(f"No edge found for condition branch: {branch}")

    return nodes_by_id[outgoing_edges[0]["target"]]


def write_node_execution(run_ref, node, result):
    execution = {
        "nodeId":         node["id"],
        "nodeLabel":      node["data"].get("label"),
        "kind":           node["data"].get("kind"),
        "status":         "completed" if not result.get("error") else "failed",
        "output":         result.get("output"),
        "outputVariable": result.get("outputVariable"),
        "error":          result.get("error"),
        "completedAt":    datetime.now(timezone.utc),
    }
    run_ref.update({
        "nodeExecutions": firestore.ArrayUnion([execution]),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    })


def apply_column_mappings(item, node_data):
    column_mappings = node_data.get("columnMappings", [])
    row_data = {}
    for mapping in column_mappings:
        field = mapping.get("field")
        column = mapping.get("column")
        if not field or not column or column == "exclude":
            continue
        if isinstance(item, dict):
            value = item.get(field)
        else:
            value = str(item) if item is not None else None
        if value is not None:
            row_data[column] = value
    return row_data


def node_references_column(node_data, column_keys):
    """
    Returns True if any string field in node_data contains @variable
    where variable matches a column key.
    """
    if not column_keys:
        return False

    column_key_set = set(column_keys)

    def check_value(value):
        if isinstance(value, str):
            refs = re.findall(r'@(\w+)', value)
            return any(ref in column_key_set for ref in refs)
        if isinstance(value, dict):
            return any(check_value(v) for v in value.values())
        if isinstance(value, list):
            return any(check_value(v) for v in value)
        return False

    return check_value(node_data)


def flush_table_to_firestore(run_ref, column_keys, output_table_rows):
    """Write the full in-memory table to Firestore as a map."""
    rows_map = {str(i): row for i, row in enumerate(output_table_rows)}
    run_ref.update({
        "outputTable": {
            "columns": column_keys,
            "rows": rows_map,
        },
    })


def build_table_variable(column_keys, output_table_rows):
    """
    Build the outputTable structure expected by the interpolator's
    is_output_table() check — canonical columns from version config,
    rows as an indexed map.
    """
    return {
        "columns": column_keys,
        "rows": {str(i): row for i, row in enumerate(output_table_rows)},
    }


def execute_with_retry(executor_fn, node_data, max_retries=3, **kwargs):
    """
    Execute a node function with:
    - Proactive delay (rateLimitDelay from node config, default 300ms for row loops)
    - Retry on 429 with exponential backoff + jitter
    - Respect Retry-After header if present
    """
    rate_limit_delay = node_data.get("rateLimitDelay", 0.3)
    if rate_limit_delay > 0:
        time.sleep(rate_limit_delay)

    for attempt in range(max_retries):
        try:
            return executor_fn(**kwargs)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                # check for Retry-After header
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")

                if retry_after:
                    wait = float(retry_after)
                else:
                    # exponential backoff + jitter
                    wait = (2 ** attempt) + random.uniform(0, 0.5)

                logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.2f}s...")
                time.sleep(wait)
            else:
                raise


# ─── Main executor ────────────────────────────────────────────────────────────

def execute_workflow(db, run_ref, run_data):
    agent_ref = run_ref.parent.parent
    version_id = run_data.get("versionId")

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
    output_type = version.get("outputType", "single")
    table_columns = version.get("tableColumns", [])
    column_keys = [col["key"] for col in table_columns]

    run_data["outputType"] = output_type

    if not nodes:
        raise Exception("Workflow has no nodes")

    nodes_by_id, edges_by_source = build_lookup_maps(nodes, edges)

    # determine start node
    resume_at_node_id   = run_data.get("resumeAtNodeId")
    resume_from_node_id = run_data.get("resumeFromNodeId")

    if resume_at_node_id:
        if resume_at_node_id not in nodes_by_id:
            raise Exception(f"resumeAtNodeId {resume_at_node_id} not found in version")
        current_node = nodes_by_id[resume_at_node_id]
        run_ref.update({"resumeAtNodeId": firestore.DELETE_FIELD})
        logger.info(f"Resuming workflow at node {resume_at_node_id}")

    elif resume_from_node_id:
        outgoing = edges_by_source.get(resume_from_node_id, [])
        if not outgoing:
            raise Exception(f"No outgoing edges from resumeFromNodeId {resume_from_node_id}")
        next_node_id = outgoing[0]["target"]
        if next_node_id not in nodes_by_id:
            raise Exception(f"resumeFromNodeId target {next_node_id} not found in version")
        current_node = nodes_by_id[next_node_id]
        run_ref.update({"resumeFromNodeId": firestore.DELETE_FIELD})
        logger.info(f"Resuming workflow after node {resume_from_node_id} → starting at {current_node['id']}")

    else:
        current_node = find_trigger_node(nodes, edges)
        logger.info(f"Starting workflow execution from trigger node: {current_node['id']}")

    variables = run_data.get("variables", {}).copy()

    # Site 1: re-hydrate table rows from Firestore so resuming runs have the latest data
    saved_rows = run_data.get("outputTable", {}).get("rows", {})
    output_table_rows = [saved_rows[k] for k in sorted(saved_rows.keys(), key=lambda x: int(x))]
    if output_table_rows:
        variables["table"] = build_table_variable(column_keys, output_table_rows)

    visited = set()

    while current_node:
        node_id = current_node["id"]
        node_kind = current_node["data"].get("kind")
        node_data = current_node["data"]

        if node_id in visited:
            raise Exception(f"Cycle detected at node {node_id}")
        visited.add(node_id)

        logger.info(f"Executing node: {node_id} ({node_kind})")

        executor_fn = NODE_EXECUTORS.get(node_kind)
        if not executor_fn:
            raise Exception(f"Unknown node kind: {node_kind}")

        # pass current table rows into variables so eval node can access them
        if output_type == "table":
            variables["_output_table_rows"] = output_table_rows

        interpolated_node = interpolate(current_node, variables)

        # ── detect row analyzer before execution ──────────────────────────────
        is_row_analyzer = (
            output_type == "table"
            and node_data.get("tableOutputColumn")
            and node_data.get("tableOutputColumn") != "exclude"
        )

        if is_row_analyzer:
            output_column = node_data["tableOutputColumn"]
            logger.info(f"[Table] Row analyzer node {node_id} — processing {len(output_table_rows)} rows into column '{output_column}'")

            result = {"output": None, "outputVariable": None}

            for row_index, row in enumerate(output_table_rows):
                row_variables = variables.copy()
                for col_key, col_value in row.items():
                    row_variables[col_key] = col_value

                interpolated_row_node = interpolate(current_node, row_variables)

                try:
                    row_result = execute_with_retry(
                        executor_fn,
                        node_data=node_data,
                        node=interpolated_row_node,
                        variables=row_variables,
                        db=db,
                        run_ref=run_ref,
                        run_data=run_data,
                    )
                except Exception as e:
                    logger.warning(f"[Table] Node {node_id} failed for row {row_index} after retries: {e}")
                    output_table_rows[row_index][output_column] = None
                    continue

                if row_result.get("error"):
                    logger.warning(f"[Table] Node {node_id} failed for row {row_index}: {row_result['error']}")
                    output_table_rows[row_index][output_column] = None
                else:
                    row_output = row_result.get("output")
                    output_table_rows[row_index][output_column] = row_output
                    run_ref.update({
                        f"outputTable.rows.{row_index}.{output_column}": row_output,
                    })

            # Site 2: after row analyzer finishes
            variables["table"] = build_table_variable(column_keys, output_table_rows)
            write_node_execution(run_ref, current_node, result)

        else:
            result = executor_fn(
                node=interpolated_node,
                variables=variables,
                db=db,
                run_ref=run_ref,
                run_data=run_data,
            )

            write_node_execution(run_ref, current_node, result)

            if result.get("pause"):
                logger.info(f"Run paused at node {node_id}")
                return

            if result.get("error"):
                raise Exception(f"Node {node_id} failed: {result['error']}")

            # if eval node ran in table mode, pick up surviving rows
            if node_kind == "eval" and "output_table_rows" in result:
                output_table_rows = result["output_table_rows"]
                flush_table_to_firestore(run_ref, column_keys, output_table_rows)  # renumber keys to stay in sync
                variables["table"] = build_table_variable(column_keys, output_table_rows)
                logger.info(f"[Eval] In-memory table updated — {len(output_table_rows)} rows remaining")

            output_value = result.get("output")
            output_variable = result.get("outputVariable")

            # ── table mode ────────────────────────────────────────────────────
            if output_type == "table":

                if isinstance(output_value, list) and len(output_value) > 0:

                    is_per_row = (
                        len(output_table_rows) > 0 and
                        node_references_column(node_data, column_keys)
                    )

                    if is_per_row:
                        logger.info(f"[Table] Per-row producer node {node_id} — expanding {len(output_table_rows)} rows")
                        expanded_rows = []
                        for existing_row in output_table_rows:
                            row_variables = variables.copy()
                            for col_key, col_value in existing_row.items():
                                row_variables[col_key] = col_value

                            interpolated_row_node = interpolate(current_node, row_variables)

                            try:
                                row_result = execute_with_retry(
                                    executor_fn,
                                    node_data=node_data,
                                    node=interpolated_row_node,
                                    variables=row_variables,
                                    db=db,
                                    run_ref=run_ref,
                                    run_data=run_data,
                                )
                            except Exception as e:
                                logger.warning(f"[Table] Per-row producer node {node_id} failed for row after retries: {e}")
                                expanded_rows.append(existing_row)
                                continue

                            row_output = row_result.get("output")
                            if isinstance(row_output, list):
                                for item in row_output:
                                    new_row = existing_row.copy()
                                    new_row.update(apply_column_mappings(item, node_data))
                                    expanded_rows.append(new_row)
                            else:
                                expanded_rows.append(existing_row)

                        output_table_rows = expanded_rows
                        logger.info(f"[Table] Expanded to {len(output_table_rows)} rows")

                    else:
                        logger.info(f"[Table] Global row producer node {node_id} — appending {len(output_value)} rows")
                        for item in output_value:
                            row_data = apply_column_mappings(item, node_data)
                            output_table_rows.append(row_data)

                    flush_table_to_firestore(run_ref, column_keys, output_table_rows)
                    # Site 3: after global/per-row producer flushes
                    variables["table"] = build_table_variable(column_keys, output_table_rows)

                else:
                    if output_variable and output_value is not None:
                        variables[output_variable] = output_value
                        run_ref.update({
                            f"variables.{output_variable}": output_value,
                            f"outputVariables.{output_variable}": output_value,
                        })

            # ── single mode ───────────────────────────────────────────────────
            else:
                if output_variable and output_value is not None:
                    variables[output_variable] = output_value
                    run_ref.update({
                        f"variables.{output_variable}": output_value,
                        f"outputVariables.{output_variable}": output_value,
                    })

        # find next node
        outgoing = edges_by_source.get(node_id, [])
        if not outgoing:
            logger.info(f"Workflow completed — no more edges from node {node_id}")
            run_ref.update({
                "status": "completed",
                "completedAt": firestore.SERVER_TIMESTAMP,
                "lockedBy": None,
            })
            return

        current_node = pick_next_node(current_node, outgoing, result, nodes_by_id)

    run_ref.update({
        "status": "completed",
        "completedAt": firestore.SERVER_TIMESTAMP,
        "lockedBy": None,
    })
    logger.info("Workflow execution complete")