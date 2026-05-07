import json

from vllm_mlx.api.models import FunctionCall, ToolCall
from vllm_mlx.api.tool_calling import (
    compact_json_dumps,
    convert_tools_for_template,
    looks_like_malformed_tool_call,
    validate_tool_calls_against_tools,
)

LONG_DESCRIPTION = (
    "Search the internal knowledge base for documents. "
    "Use this when the user asks for precise factual lookup, source-backed "
    "answers, or recent project-specific context. " * 6
)


def _sample_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": LONG_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": LONG_DESCRIPTION,
                        },
                        "limit": {
                            "type": "integer",
                            "enum": [1, 3, 5],
                            "description": "Maximum result count.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _tool_call(name: str, arguments: dict | str) -> ToolCall:
    return ToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(
            name=name,
            arguments=(
                json.dumps(arguments) if isinstance(arguments, dict) else arguments
            ),
        ),
    )


def test_compact_tools_preserve_schema_shape_and_trim_descriptions():
    original = convert_tools_for_template(_sample_tools(), compact=False)
    compact = convert_tools_for_template(_sample_tools(), compact=True)

    original_func = original[0]["function"]
    compact_func = compact[0]["function"]

    assert compact_func["name"] == original_func["name"]
    assert compact_func["parameters"]["required"] == ["query"]
    assert compact_func["parameters"]["properties"]["limit"]["enum"] == [1, 3, 5]
    assert compact_func["parameters"]["additionalProperties"] is False
    assert len(compact_func["description"]) < len(original_func["description"])
    assert (
        len(compact_func["parameters"]["properties"]["query"]["description"])
        < len(original_func["parameters"]["properties"]["query"]["description"])
    )
    assert len(compact_json_dumps(compact)) < len(json.dumps(original))


def test_compact_json_dumps_removes_redundant_whitespace():
    data = {"b": [1, 2], "a": {"x": "y"}}

    rendered = compact_json_dumps(data)

    assert rendered == '{"b":[1,2],"a":{"x":"y"}}'


def test_full_schema_validation_accepts_valid_tool_call():
    result = validate_tool_calls_against_tools(
        [_tool_call("search_docs", {"query": "mlx", "limit": 3})],
        _sample_tools(),
    )

    assert result.ok is True
    assert result.error is None


def test_full_schema_validation_rejects_missing_required_param():
    result = validate_tool_calls_against_tools(
        [_tool_call("search_docs", {"limit": 3})],
        _sample_tools(),
    )

    assert result.ok is False
    assert "required" in result.error.lower()


def test_full_schema_validation_rejects_enum_mismatch():
    result = validate_tool_calls_against_tools(
        [_tool_call("search_docs", {"query": "mlx", "limit": 9})],
        _sample_tools(),
    )

    assert result.ok is False
    assert "one of" in result.error.lower()


def test_full_schema_validation_rejects_unknown_tool():
    result = validate_tool_calls_against_tools(
        [_tool_call("unknown_tool", {"query": "mlx"})],
        _sample_tools(),
    )

    assert result.ok is False
    assert "unknown" in result.error.lower()


def test_full_schema_validation_rejects_invalid_json_arguments():
    result = validate_tool_calls_against_tools(
        [_tool_call("search_docs", '{"query":')],
        _sample_tools(),
    )

    assert result.ok is False
    assert "valid json" in result.error.lower()


def test_malformed_tool_detection_is_marker_based():
    assert looks_like_malformed_tool_call("<tool_call>{bad json")
    assert looks_like_malformed_tool_call('[Calling tool: search_docs({"q":')
    assert not looks_like_malformed_tool_call("I can answer without a tool.")
