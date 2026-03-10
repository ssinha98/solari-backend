import logging
import re
from workflow.evals import llm_judge

logger = logging.getLogger(__name__)


# ─── Regex eval ───────────────────────────────────────────────────────────────

def evaluate_regex(input_text: str, pattern: str) -> dict:
    """
    Check if input_text matches the given regex pattern.
    Passes if a match is found.
    """
    try:
        match = re.search(pattern, input_text)
        passed = match is not None
        return {
            "pass": passed,
            "score": 1.0 if passed else 0.0,
            "reason": f"Pattern '{pattern}' {'matched' if passed else 'did not match'}",
            "branch": "pass" if passed else "fail",
        }
    except re.error as e:
        raise Exception(f"Invalid regex pattern '{pattern}': {e}")


# ─── Table row deletion (mirrors review_delete_rows endpoint logic) ────────────

def delete_rows_from_firestore(run_ref, row_indices: list):
    """
    Delete specific rows from outputTable.rows in Firestore.
    Mirrors the logic in the review_delete_rows endpoint.
    """
    run_snap = run_ref.get()
    if not run_snap.exists:
        logger.error("Run doc not found when trying to delete eval-failed rows")
        return

    output_table = run_snap.to_dict().get("outputTable", {})
    rows = output_table.get("rows", {})

    for idx in row_indices:
        key = str(idx)
        if key in rows:
            del rows[key]
            logger.info(f"[Eval] Deleted row {idx} from outputTable")
        else:
            logger.warning(f"[Eval] Row {idx} not found in outputTable — skipping")

    output_table["rows"] = rows
    run_ref.update({"outputTable": output_table})


# ─── Main executor ────────────────────────────────────────────────────────────

def execute(node: dict, variables: dict, db, run_ref, run_data: dict) -> dict:
    """
    Eval node executor.

    Reads evalType from node data and routes to the correct eval implementation.
    Always returns a branch of "pass" or "fail" for graph traversal.

    In table mode (when evalInputColumn is set), iterates output_table_rows
    row by row and either deletes or marks failing rows based on evalFailBehavior.

    In single mode, evaluates the most recently written variable.
    """
    node_data = node.get("data", {})
    label = node_data.get("label", "Eval")
    eval_type = node_data.get("evalType", "llm_judge")
    user_id = run_data.get("triggeredBy")
    eval_input_column = node_data.get("evalInputColumn")  # set in table mode
    eval_fail_behavior = node_data.get("evalFailBehavior", "mark")  # "delete" or "mark"
    output_type = run_data.get("outputType", "single")

    # ── Table mode: row-by-row eval ───────────────────────────────────────────
    if output_type == "table" and eval_input_column:
        output_table_rows = variables.get("_output_table_rows", [])

        if not output_table_rows:
            logger.warning(f"Eval node '{label}' in table mode but no rows found")
            return {
                "output": "pass",
                "outputVariable": None,
                "branch": "pass",
                "pause": False,
                "error": None,
                "output_table_rows": output_table_rows,
            }

        rows_to_delete = []
        surviving_rows = []

        for row_index, row in enumerate(output_table_rows):
            input_text = str(row.get(eval_input_column, ""))

            if not input_text:
                logger.warning(f"[Eval] Row {row_index} has no value for column '{eval_input_column}' — marking as fail")
                if eval_fail_behavior == "delete":
                    rows_to_delete.append(row_index)
                else:
                    row["eval_failed"] = True
                    row["eval_reason"] = f"No value for column '{eval_input_column}'"
                    surviving_rows.append(row)
                continue

            try:
                if eval_type == "llm_judge":
                    result = llm_judge.evaluate(
                        input_text=input_text,
                        criteria=node_data.get("evalCriteria", ""),
                        model=node_data.get("evalModel", "gpt-4o"),
                        system_prompt=node_data.get("evalSystemPrompt", ""),
                        threshold=float(node_data.get("evalThreshold", 0.7)),
                        user_id=user_id,
                    )
                elif eval_type == "regex":
                    result = evaluate_regex(
                        input_text=input_text,
                        pattern=node_data.get("evalCriteria", ""),
                    )
                else:
                    raise Exception(f"Unsupported eval type: {eval_type}")

                passed = result["pass"]
                logger.info(f"[Eval] Row {row_index} — {'pass' if passed else 'fail'} ({result.get('reason', '')})")

                if passed:
                    surviving_rows.append(row)
                else:
                    if eval_fail_behavior == "delete":
                        rows_to_delete.append(row_index)
                    else:
                        row["eval_failed"] = True
                        row["eval_reason"] = result.get("reason", "")
                        surviving_rows.append(row)

            except Exception as e:
                logger.warning(f"[Eval] Row {row_index} eval error: {e} — treating as fail")
                if eval_fail_behavior == "delete":
                    rows_to_delete.append(row_index)
                else:
                    row["eval_failed"] = True
                    row["eval_reason"] = str(e)
                    surviving_rows.append(row)

        # delete failing rows from Firestore in one write
        if rows_to_delete:
            logger.info(f"[Eval] Deleting {len(rows_to_delete)} failing rows from Firestore: {rows_to_delete}")
            delete_rows_from_firestore(run_ref, rows_to_delete)

        # if marking, flush updated rows (with eval_failed fields) to Firestore
        if eval_fail_behavior == "mark" and surviving_rows:
            rows_map = {str(i): row for i, row in enumerate(surviving_rows)}
            run_ref.update({"outputTable.rows": rows_map})

        logger.info(f"[Eval] Table eval complete — {len(surviving_rows)} rows surviving, {len(rows_to_delete)} deleted")

        return {
            "output": "pass",
            "outputVariable": None,
            "branch": "pass",
            "pause": False,
            "error": None,
            "output_table_rows": surviving_rows,  # graph picks this up to update in-memory rows
        }

    # ── Single mode: evaluate one variable ───────────────────────────────────
    eval_input_variable = node_data.get("evalInputVariable")
    if eval_input_variable and eval_input_variable in variables:
        input_text = str(variables[eval_input_variable])
    else:
        input_text = str(list(variables.values())[-1]) if variables else ""

    if not input_text:
        return {
            "output": "fail — no input to evaluate",
            "outputVariable": None,
            "branch": "fail",
            "pause": False,
            "error": "No input text found to evaluate",
        }

    logger.info(f"Eval node '{label}' running {eval_type} eval")

    try:
        if eval_type == "llm_judge":
            result = llm_judge.evaluate(
                input_text=input_text,
                criteria=node_data.get("evalCriteria", ""),
                model=node_data.get("evalModel", "gpt-4o"),
                system_prompt=node_data.get("evalSystemPrompt", ""),
                threshold=float(node_data.get("evalThreshold", 0.7)),
                user_id=user_id,
            )
        elif eval_type == "regex":
            result = evaluate_regex(
                input_text=input_text,
                pattern=node_data.get("evalCriteria", ""),
            )
        else:
            raise Exception(f"Unsupported eval type: {eval_type}")

        return {
            "output": result["branch"],
            "outputVariable": None,
            "branch": result["branch"],
            "pause": False,
            "error": None,
            "evalResult": {
                "pass": result["pass"],
                "score": result.get("score"),
                "reason": result.get("reason"),
                "evalType": eval_type,
            },
        }

    except Exception as e:
        logger.error(f"Eval node '{label}' failed: {e}", exc_info=True)
        return {
            "output": "fail",
            "outputVariable": None,
            "branch": "fail",
            "pause": False,
            "error": str(e),
        }