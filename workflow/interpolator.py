import re
import copy
import logging

logger = logging.getLogger(__name__)

# matches @variable_name references
VARIABLE_PATTERN = re.compile(r'@(\w+)')


def interpolate_string(text: str, variables: dict) -> str:
    """
    Replace all {{variable_name}} references in a string with values
    from the variables map. Logs a warning if a variable is not found.
    """
    def replace_match(match):
        var_name = match.group(1)
        if var_name in variables:
            value = variables[var_name]
            # convert non-string values to string for interpolation
            if isinstance(value, (dict, list)):
                import json
                return json.dumps(value)
            return str(value)
        else:
            logger.warning(f"Variable '{var_name}' not found in run variables — leaving as-is")
            return match.group(0)  # leave {{variable_name}} unchanged if not found

    return VARIABLE_PATTERN.sub(replace_match, text)


def interpolate_value(value, variables: dict):
    """
    Recursively interpolate a value — handles strings, dicts, and lists.
    Non-string primitives (int, bool, None) are returned as-is.
    """
    if isinstance(value, str):
        return interpolate_string(value, variables)

    if isinstance(value, dict):
        return { k: interpolate_value(v, variables) for k, v in value.items() }

    if isinstance(value, list):
        return [ interpolate_value(item, variables) for item in value ]

    # int, float, bool, None — return unchanged
    return value


def interpolate(node: dict, variables: dict) -> dict:
    """
    Return a deep copy of the node with all {{variable_name}} references
    in node.data replaced with their values from the variables map.

    Only interpolates the data field — position, id, type are left unchanged.
    """
    interpolated = copy.deepcopy(node)
    interpolated["data"] = interpolate_value(node.get("data", {}), variables)
    return interpolated