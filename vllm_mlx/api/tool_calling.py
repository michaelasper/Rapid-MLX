# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

Supports parsing tool calls from multiple model formats:
- Qwen:
- Llama:

Also includes structured output (JSON Schema) utilities:
- parse_json_output: Extract JSON from model output
- validate_json_schema: Validate JSON against a schema
"""

import json
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError, validate

from .models import FunctionCall, ResponseFormat, ToolCall

logger = logging.getLogger(__name__)

_DESCRIPTION_MAX_CHARS = 240
_EXAMPLE_MAX_CHARS = 160


@dataclass(frozen=True)
class ToolValidationResult:
    """Full-schema validation result for generated tool calls."""

    ok: bool
    error: str | None = None


def compact_json_dumps(value: Any) -> str:
    """Render JSON without redundant whitespace."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical_json_dumps(value: Any) -> str:
    """Render JSON deterministically for cache keys."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _extract_tool_call_parts(tool_call: Any) -> tuple[str, Any] | None:
    if isinstance(tool_call, dict) and "function" not in tool_call:
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", "{}")
        if not isinstance(name, str) or not name.strip():
            return None
        return name.strip(), arguments

    func = (
        tool_call.function
        if hasattr(tool_call, "function")
        else tool_call.get("function", {})
        if isinstance(tool_call, dict)
        else {}
    )
    name = func.name if hasattr(func, "name") else func.get("name", "")
    arguments = (
        func.arguments
        if hasattr(func, "arguments")
        else func.get("arguments", "{}")
        if isinstance(func, dict)
        else "{}"
    )
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip(), arguments


def canonical_tool_call_key(tool_call: Any) -> str | None:
    """Build an exact semantic cache key for a generated tool call.

    The key intentionally normalizes only JSON structure and object-key order.
    It does not fuzzy-match, coerce path strings, or infer missing arguments:
    false cache hits are worse than misses for speculative tool execution.
    """

    parts = _extract_tool_call_parts(tool_call)
    if parts is None:
        return None
    name, arguments = parts
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    return _canonical_json_dumps({"name": name, "arguments": arguments})


def _model_or_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(by_alias=True, exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    data: dict[str, Any] = {}
    for attr in ("type", "function", "name", "description", "parameters"):
        attr_value = getattr(value, attr, None)
        if attr_value is None:
            continue
        is_mock_value = attr_value.__class__.__module__ == "unittest.mock"
        if attr == "function":
            data[attr] = attr_value
            continue
        if callable(attr_value):
            continue
        if is_mock_value and attr != "function":
            continue
        data[attr] = attr_value
    return data


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _trim_text(text: str, max_chars: int) -> str:
    text = _collapse_whitespace(text)
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1].rsplit(" ", 1)[0].rstrip()
    return f"{cut}…" if cut else text[: max_chars - 1] + "…"


def _compact_schema_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key == "description":
            return _trim_text(value, _DESCRIPTION_MAX_CHARS)
        if key in ("example", "examples"):
            return _trim_text(value, _EXAMPLE_MAX_CHARS)
        return value
    if isinstance(value, list):
        return [_compact_schema_value(item, key=key) for item in value]
    if isinstance(value, dict):
        return {
            child_key: _compact_schema_value(child_value, key=child_key)
            for child_key, child_value in value.items()
        }
    return value


def _normalize_tool_definition(tool: Any, *, compact: bool) -> dict[str, Any] | None:
    tool_dict = _model_or_dict(tool)
    tool_type = tool_dict.get("type")
    tool_func = tool_dict.get("function")
    if tool_type != "function" or not tool_func:
        return None

    func = _model_or_dict(tool_func)
    func_name = func.get("name", "")
    func_desc = func.get("description", "")
    func_params = func.get("parameters", {"type": "object", "properties": {}})

    if compact:
        func_desc = _trim_text(str(func_desc), _DESCRIPTION_MAX_CHARS)
        func_params = _compact_schema_value(func_params)

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_desc,
            "parameters": func_params,
        },
    }


def looks_like_malformed_tool_call(text: str) -> bool:
    """Detect outputs that appear to intend a tool call but failed parsing."""

    if not text:
        return False
    structural_markers = (
        "<tool_call",
        "</tool_call",
        "<function=",
        "<|python_tag|>",
        "[Calling tool:",
    )
    if any(marker in text for marker in structural_markers):
        return True
    return "{" in text and (
        ('"name"' in text and '"arguments"' in text) or '"parameters"' in text
    )


def validate_tool_calls_against_tools(
    tool_calls: list[Any] | None,
    tools: list[Any] | None,
) -> ToolValidationResult:
    """Validate generated tool calls against the full original tool schemas."""

    if not tool_calls:
        return ToolValidationResult(ok=True)
    if not tools:
        return ToolValidationResult(ok=False, error="Tool call emitted without tools")

    tool_defs: dict[str, dict[str, Any]] = {}
    for tool in tools:
        normalized = _normalize_tool_definition(tool, compact=False)
        if not normalized:
            continue
        func = normalized["function"]
        name = func.get("name")
        if name:
            tool_defs[name] = func

    for tc in tool_calls:
        func = tc.function if hasattr(tc, "function") else tc.get("function", {})
        func_name = func.name if hasattr(func, "name") else func.get("name", "")
        args_raw = (
            func.arguments
            if hasattr(func, "arguments")
            else func.get("arguments", "{}")
        )

        if func_name not in tool_defs:
            return ToolValidationResult(
                ok=False,
                error=f"Unknown tool call '{func_name}'",
            )

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, ValueError) as e:
            return ToolValidationResult(
                ok=False,
                error=f"Tool call '{func_name}' arguments is not valid JSON: {e}",
            )

        if not isinstance(args, dict):
            return ToolValidationResult(
                ok=False,
                error=f"Tool call '{func_name}' arguments must be a JSON object",
            )

        schema = tool_defs[func_name].get(
            "parameters", {"type": "object", "properties": {}}
        )
        try:
            validate(instance=args, schema=schema)
        except ValidationError as e:
            return ToolValidationResult(
                ok=False,
                error=f"Tool call '{func_name}' schema validation failed: {e.message}",
            )

    return ToolValidationResult(ok=True)


def _is_tool_call_json(obj: dict) -> bool:
    """
    Check if a JSON object looks like a tool call.

    A tool call must have:
    - "name" key with a string value (function name)
    - "arguments" key (the function arguments)

    This prevents false positives where regular JSON like {"name": "John", "age": 30}
    would be incorrectly parsed as a tool call.

    Args:
        obj: JSON object to check

    Returns:
        True if object appears to be a tool call
    """
    if not isinstance(obj, dict):
        return False

    # Must have both "name" and "arguments" keys
    if "name" not in obj or "arguments" not in obj:
        return False

    # "name" must be a non-empty string (function name)
    if not isinstance(obj["name"], str) or not obj["name"].strip():
        return False

    # "arguments" must be a dict or string
    args = obj["arguments"]
    if not isinstance(args, (dict, str)):
        return False

    return True


def _parse_raw_json_tool_calls(text: str) -> list[dict] | None:
    """
    Parse raw JSON tool calls from model output.

    Handles:
    - Single JSON object: {"name": "func", "arguments": {...}}
    - Multiple objects separated by commas: {...}, {...}
    - JSON array: [{...}, {...}]

    Only objects with BOTH "name" AND "arguments" keys are considered tool calls.
    This prevents false positives with regular JSON objects.

    Args:
        text: Raw model output text

    Returns:
        List of tool call dicts with 'name' and 'arguments', or None if no valid tool calls found
    """
    if not text:
        return None

    text = text.strip()

    # Try JSON array first
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all(
                _is_tool_call_json(item) for item in parsed
            ):
                return [
                    {"name": item["name"], "arguments": item.get("arguments", {})}
                    for item in parsed
                ]
        except json.JSONDecodeError:
            pass

    # Find JSON objects with balanced braces
    tool_calls = []
    depth = 0
    start = None

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                json_str = text[start : i + 1]
                try:
                    obj = json.loads(json_str)
                    # Only consider as tool call if it has both "name" AND "arguments"
                    if _is_tool_call_json(obj):
                        tool_calls.append(
                            {"name": obj["name"], "arguments": obj.get("arguments", {})}
                        )
                except json.JSONDecodeError:
                    pass
                start = None

    return tool_calls if tool_calls else None


def parse_tool_calls(
    text: str, request: dict[str, Any] | None = None
) -> tuple[str, list[ToolCall] | None]:
    """
    Parse tool calls from model output.

    Supports multiple formats:
    - Qwen3 bracket: [Calling tool: function_name({...})]
    - Qwen:
    - Llama:
    - Nemotron:
    - Raw JSON: {"name": "...", "arguments": {...}} (single or multiple)

    Args:
        text: Raw model output text

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    tool_calls = []
    cleaned_text = text

    # Pattern for Qwen3 bracket-style: [Calling tool: function_name({...})]
    bracket_pattern = r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]"
    bracket_matches = re.findall(bracket_pattern, text, re.DOTALL)

    for name, args_str in bracket_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove bracket tool calls from cleaned text
    if bracket_matches:
        cleaned_text = re.sub(
            r"\[Calling tool:\s*\w+\(\{.*?\}\)\]", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Nemotron-style:
    # Format 1: <tool_call><function=name><parameter=key>val</parameter></function></tool_call>
    # Format 2: <toolcall>func_name\n<parameter=key>value</parameter>...</toolcall>
    nemotron_pattern = r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>"
    nemotron_matches = re.findall(nemotron_pattern, text, re.DOTALL)
    if not nemotron_matches:
        nemotron_pattern = r"<toolcall>\s*(\w+)\s*\n(.*?)</toolcall>"
        nemotron_matches = re.findall(nemotron_pattern, text, re.DOTALL)

    for name, params_block in nemotron_matches:
        # Parse parameters from <parameter=name>value</parameter> format
        param_pattern = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
        params = re.findall(param_pattern, params_block, re.DOTALL)
        arguments = {}
        for p_name, p_value in params:
            val = p_value.strip()
            try:
                arguments[p_name.strip()] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                arguments[p_name.strip()] = val

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name.strip(), arguments=json.dumps(arguments)
                ),
            )
        )

    # Remove Nemotron tool call tags from cleaned text
    if nemotron_matches:
        cleaned_text = re.sub(
            r"<tool_call>\s*<function=\w+>.*?</function>\s*</tool_call>",
            "",
            cleaned_text,
            flags=re.DOTALL,
        )
        cleaned_text = re.sub(
            r"<toolcall>\s*\w+\s*\n.*?</toolcall>",
            "",
            cleaned_text,
            flags=re.DOTALL,
        )
        cleaned_text = cleaned_text.strip()

    # Pattern for Qwen-style tool calls:
    qwen_pattern = r"\x1b\[3m\s*(\{.*?\})\s*\x1b\[0m"
    qwen_matches = re.findall(qwen_pattern, cleaned_text, re.DOTALL)

    for match in qwen_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    # Remove Qwen tool call tags from cleaned text
    if qwen_matches:
        cleaned_text = re.sub(
            r"\x1b\[3m\s*\{.*?\}\s*\x1b\[0m", "", cleaned_text, flags=re.DOTALL
        ).strip()

    # Pattern for Llama-style:
    # Format 1: <function=name>{"arg": "val"}</function>
    # Format 2: <|python_tag|>{"name": "func", "parameters": {...}}
    llama_pattern = r"<function=(\w+)>\s*(\{.*?\})\s*</function>"
    llama_matches = re.findall(llama_pattern, cleaned_text, re.DOTALL)
    if not llama_matches:
        llama_pattern = r'<\|python_tag\|>\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\s*\}'
        llama_matches = re.findall(llama_pattern, cleaned_text, re.DOTALL)

    for name, args_str in llama_matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name.strip(),
                        arguments=(
                            json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    ),
                )
            )
        except json.JSONDecodeError:
            continue

    if llama_matches:
        cleaned_text = re.sub(
            r"<function=\w+>\s*\{.*?\}\s*</function>",
            "",
            cleaned_text,
            flags=re.DOTALL,
        )
        cleaned_text = re.sub(
            r'<\|python_tag\|>\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}\s*\}',
            "",
            cleaned_text,
            flags=re.DOTALL,
        )
        cleaned_text = cleaned_text.strip()

    # Note: We keep  tags for reasoning models
    # The user may want to see the model's reasoning process

    # Fallback: Raw JSON tool calls (lowest priority)
    # Only try if no other formats matched
    if not tool_calls:
        raw_json_calls = _parse_raw_json_tool_calls(cleaned_text)
        if raw_json_calls:
            for call_data in raw_json_calls:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=call_data["name"],
                            arguments=(
                                json.dumps(call_data["arguments"])
                                if isinstance(call_data["arguments"], dict)
                                else str(call_data["arguments"])
                            ),
                        ),
                    )
                )
            # Clean the JSON and surrounding tags from text
            cleaned_text = re.sub(r"</?tool_call>", "", cleaned_text)
            cleaned_text = re.sub(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

    return cleaned_text, tool_calls if tool_calls else None


def convert_tools_for_template(
    tools: list | None,
    *,
    compact: bool = True,
) -> list[dict] | None:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        normalized = _normalize_tool_definition(tool, compact=compact)
        if normalized:
            converted.append(normalized)

    return converted if converted else None


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    if tool_call.function is None:
        raise ValueError("ToolCall has no function attribute")
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================


def validate_json_schema(data: Any, schema: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate JSON data against a JSON Schema.

    Args:
        data: The JSON data to validate (dict, list, etc.)
        schema: JSON Schema specification

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data matches schema
        - error_message: Error description if invalid, None if valid
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    Extract JSON from model output text.

    Tries multiple strategies:
    1. Parse entire text as JSON
    2. Extract JSON from markdown code blocks
    3. Find JSON object/array in text

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON data, or None if no valid JSON found
    """
    text = text.strip()

    # Strategy 1: Try to parse entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object or array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r"(\{(?:[^{}]|\{[^{}]*\})*\})",  # Match balanced braces
        r"(\[(?:[^\[\]]|\[[^\[\]]*\])*\])",  # Match balanced brackets
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return None


def parse_json_output(
    text: str, response_format: ResponseFormat | dict[str, Any] | None = None
) -> tuple[str, dict[str, Any] | None, bool, str | None]:
    """
    Parse JSON from model output when response_format is set.

    Args:
        text: Raw model output text
        response_format: ResponseFormat specification (optional)
            - If type="json_object", extracts any valid JSON
            - If type="json_schema", extracts and validates against schema

    Returns:
        Tuple of (cleaned_text, parsed_json, is_valid, error_message)
        - cleaned_text: Original text (preserved for reference)
        - parsed_json: Extracted JSON data, or None if extraction failed
        - is_valid: True if JSON is valid (and matches schema if specified)
        - error_message: Error description if invalid, None if valid
    """
    # Handle None or text format - just return original
    if response_format is None:
        return text, None, True, None

    # Normalize response_format to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})

        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def build_json_system_prompt(
    response_format: ResponseFormat | dict[str, Any] | None = None,
) -> str | None:
    """
    Build a system prompt instruction for JSON output.

    For models without native JSON mode support, this adds instructions
    to the prompt to encourage proper JSON formatting.

    Args:
        response_format: ResponseFormat specification

    Returns:
        System prompt instruction string, or None if not needed
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type == "text":
        return None

    if format_type == "json_object":
        return (
            "⚠️ JSON OUTPUT REQUIRED ⚠️\n\n"
            "You MUST respond with ONLY valid JSON.\n\n"
            "RULES:\n"
            "- Start response with { or [\n"
            "- NO text before or after JSON\n"
            "- NO thinking or explanations\n"
            "- NO markdown code blocks (```)\n"
            "- ONLY the raw JSON\n"
            "- If the user asks for a list/array, respond with a JSON array []\n"
            "- If the user asks for multiple items, include ALL requested items\n"
            "- Follow the exact structure/keys the user specifies"
        )

    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema", {})
        schema = json_schema_spec.get("schema", {})
        name = json_schema_spec.get("name", "response")
        description = json_schema_spec.get("description", "")
        strict = json_schema_spec.get("strict", False)

        # If strict mode is enabled and guided generation is available,
        # the server should use guided decoding instead of prompt injection
        if strict:
            # Return stronger instruction for strict mode
            prompt = (
                f"⚠️ STRICT JSON OUTPUT REQUIRED ⚠️\n\n"
                f"You MUST respond with ONLY a valid JSON object matching the '{name}' schema.\n"
            )
            if description:
                prompt += f"Purpose: {description}\n"
            try:
                schema_str = json.dumps(schema, indent=2)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize JSON schema: {e}")
                schema_str = str(schema)
            prompt += (
                f"\nJSON Schema:\n```json\n{schema_str}\n```\n\n"
                "STRICT RULES:\n"
                "- Start response with {{ or [\n"
                "- NO text before or after JSON\n"
                "- NO thinking, reasoning, or explanations\n"
                "- NO markdown code blocks\n"
                "- ONLY the JSON object"
            )
            return prompt

        # Standard (non-strict) mode - still strong but less aggressive
        prompt = (
            f"⚠️ JSON OUTPUT REQUIRED ⚠️\n\n"
            f"Respond with a valid JSON object matching the '{name}' schema.\n"
        )
        if description:
            prompt += f"Purpose: {description}\n"
        try:
            schema_str = json.dumps(schema, indent=2)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize JSON schema: {e}")
            schema_str = str(schema)
        prompt += (
            f"\nJSON Schema:\n```json\n{schema_str}\n```\n\n"
            "RULES:\n"
            "- Start response with { or [\n"
            "- NO text before or after JSON\n"
            "- NO markdown code blocks\n"
            "- ONLY the JSON object"
        )
        return prompt

    return None


def extract_json_schema_for_guided(
    response_format: ResponseFormat | dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Extract JSON schema from response_format for guided generation.

    Returns the schema dict if response_format specifies json_schema type,
    otherwise returns None.

    Args:
        response_format: ResponseFormat specification

    Returns:
        JSON schema dict or None
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type != "json_schema":
        return None

    json_schema_spec = rf_dict.get("json_schema", {})
    schema = json_schema_spec.get("schema", {})

    if not schema:
        return None

    return schema
