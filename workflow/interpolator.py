import re
import copy
import json
import logging

logger = logging.getLogger(__name__)

# matches @variable_name references
VARIABLE_PATTERN = re.compile(r'@(\w+)')


def is_output_table(value) -> bool:
    """Check if a value looks like an outputTable structure (columns + rows)."""
    return (
        isinstance(value, dict)
        and "columns" in value
        and "rows" in value
        and isinstance(value["columns"], list)
    )


def table_to_html(output_table: dict) -> str:
    """Convert an outputTable (columns + rows) to an inline-styled HTML table for email."""
    columns = output_table["columns"]
    rows = output_table["rows"]

    header_cells = "".join(
        f'<th style="padding:8px 12px;background:#f3f4f6;border:1px solid #ddd;text-align:left;font-family:sans-serif;font-size:14px;">{col}</th>'
        for col in columns
    )

    body_rows = ""
    for i, row in enumerate(rows.values()):
        bg = "#ffffff" if i % 2 == 0 else "#fafafa"  # alternating row shading
        cells = "".join(
            f'<td style="padding:8px 12px;border:1px solid #ddd;font-family:sans-serif;font-size:14px;background:{bg};">{row.get(col, "")}</td>'
            for col in columns
        )
        body_rows += f"<tr>{cells}</tr>"

    return (
        '<table style="border-collapse:collapse;width:100%;margin:16px 0;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        "</table>"
    )


def interpolate_string(text: str, variables: dict) -> str:
    """
    Replace all @variable_name references in a string with values
    from the variables map. Logs a warning if a variable is not found.
    """
    def replace_match(match):
        var_name = match.group(1)
        if var_name in variables:
            value = variables[var_name]
            # output tables render as HTML
            if is_output_table(value):
                return table_to_html(value)
            # convert non-string values to string for interpolation
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value)
        else:
            logger.warning(f"Variable '{var_name}' not found in run variables — leaving as-is")
            return match.group(0)  # leave @variable_name unchanged if not found

    return VARIABLE_PATTERN.sub(replace_match, text)


def interpolate_value(value, variables: dict):
    """
    Recursively interpolate a value — handles strings, dicts, and lists.
    Non-string primitives (int, bool, None) are returned as-is.
    """
    if isinstance(value, str):
        return interpolate_string(value, variables)

    if isinstance(value, dict):
        return {k: interpolate_value(v, variables) for k, v in value.items()}

    if isinstance(value, list):
        return [interpolate_value(item, variables) for item in value]

    # int, float, bool, None — return unchanged
    return value


def interpolate(node: dict, variables: dict) -> dict:
    """
    Return a deep copy of the node with all @variable_name references
    in node.data replaced with their values from the variables map.

    Only interpolates the data field — position, id, type are left unchanged.
    """
    interpolated = copy.deepcopy(node)
    interpolated["data"] = interpolate_value(node.get("data", {}), variables)
    return interpolated