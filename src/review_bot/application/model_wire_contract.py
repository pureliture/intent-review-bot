"""Pure pinned Beam wire representation shared by transport and replay.

No HTTP, admission, provider state or logical request mutation belongs here.
"""
from copy import deepcopy
from hashlib import sha256
import json

BEAM_CLOUD_MODEL = "beam:cloud"
BEAM_CONTEXT_TOKENS = 1048576
BEAM_MAX_OUTPUT_TOKENS = 32678


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")


def compact_schema(schema):
    """Intern identical schema nodes, leaving the logical schema untouched.

    Local references have no siblings. Expanding them reproduces the exact
    original schema; existing reference dialects are deliberately left alone.
    """
    if not isinstance(schema, dict):
        return schema
    maps = {"properties", "patternProperties", "dependentSchemas"}
    singles = {"items", "additionalProperties", "contains", "propertyNames", "if", "then", "else", "not"}
    arrays = {"allOf", "anyOf", "oneOf", "prefixItems"}
    counts, values = {}, {}

    def children(node, visit):
        if not isinstance(node, dict):
            return deepcopy(node)
        result = deepcopy(node)
        for key, value in node.items():
            if key in maps:
                result[key] = {name: visit(child) for name, child in value.items()}
            elif key in singles and isinstance(value, (dict, bool)):
                result[key] = visit(value)
            elif key in arrays:
                result[key] = [visit(child) for child in value]
        return result

    def count(node):
        if isinstance(node, dict):
            if any(key in node for key in ("$ref", "$defs", "definitions", "$id", "$anchor", "$dynamicRef", "$dynamicAnchor")):
                raise ValueError("existing schema reference dialect")
            key = _encode(node)
            counts[key] = counts.get(key, 0) + 1
            values[key] = node
            children(node, count)
        return node

    try:
        count(schema)
    except ValueError:
        return schema
    keys = sorted((key for key in counts if counts[key] > 1), key=lambda key: (-len(key), key))
    names = {key: "s" + str(index) for index, key in enumerate(keys)}

    def build(node, root=False):
        if isinstance(node, dict) and not root and _encode(node) in names:
            return {"$ref": "#/$defs/" + names[_encode(node)]}
        return children(node, build)

    result = build(schema, True)
    result["$defs"] = {names[key]: build(values[key], True) for key in keys}
    # Prune by exact byte cost, not an arbitrary schema/item size threshold.
    while True:
        for name, target in result["$defs"].items():
            def inline(node):
                return deepcopy(target) if node == {"$ref": "#/$defs/" + name} else children(node, inline)
            candidate = inline(result)
            # $defs is a map of schema nodes, outside the original dialect.
            candidate["$defs"] = {key: inline(value) for key, value in result["$defs"].items() if key != name}
            if len(_encode(candidate)) <= len(_encode(result)):
                result = candidate
                break
        else:
            break
    if not result["$defs"]:
        result.pop("$defs")
    return result if len(_encode(result)) < len(_encode(schema)) else schema



def beam_wire_payload(request):
    schema = compact_schema(request.response_schema)
    contract = "\n\nReturn exactly one JSON object matching this entire response schema. Do not add prose or fields.\n" + _encode(schema).decode("utf-8")
    return {"model": request.model, "messages": [
        {"role": "system", "content": request.system_prompt + contract},
        {"role": "user", "content": request.user_content}],
        "format": schema, "stream": False, "think": False, "truncate": False, "shift": False,
        "options": {"num_ctx": BEAM_CONTEXT_TOKENS,
            "num_predict": request.max_output_tokens if request.max_output_tokens is not None else BEAM_MAX_OUTPUT_TOKENS,
            "temperature": 0, "seed": 0}}


def beam_meter_wire_sha256(request):
    payload = beam_wire_payload(request)
    payload["options"]["num_predict"] = 1
    return sha256(_encode(payload)).hexdigest()
