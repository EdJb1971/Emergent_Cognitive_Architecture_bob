import ast
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

_UNICODE_QUOTE_TRANS = str.maketrans({
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
})

class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def _normalize_unicode_quotes(text: str) -> str:
    """Normalize curly quotes so downstream JSON parsing stays predictable."""
    return text.translate(_UNICODE_QUOTE_TRANS)


_INCOMPLETE_TAIL_PATTERNS = (
    re.compile(r',\s*"[^"]*"\s*$'),
    re.compile(r',\s*"[^"]*"\s*:\s*$'),
)


def _remove_incomplete_tail(text: str) -> str:
    """Drop trailing key/value fragments that cannot form valid JSON."""
    trimmed = text.rstrip()
    while trimmed:
        removed = False
        for pattern in _INCOMPLETE_TAIL_PATTERNS:
            match = pattern.search(trimmed)
            if match and match.end() == len(trimmed):
                trimmed = trimmed[: match.start()].rstrip()
                removed = True
                break
        if not removed:
            break
    return trimmed


def _trim_to_valid_json_suffix(text: str) -> str:
    """Trim trailing characters until the string ends with a valid JSON terminator."""
    valid_suffix_chars = set('}]"0123456789eE')
    trimmed = _remove_incomplete_tail(text.rstrip())
    while trimmed and trimmed[-1] not in valid_suffix_chars:
        trimmed = trimmed[:-1].rstrip()
    return trimmed


def _balance_json_brackets(text: str) -> str:
    """Balance unmatched braces/brackets by appending closing characters."""
    brace_diff = text.count('{') - text.count('}')
    bracket_diff = text.count('[') - text.count(']')

    if brace_diff > 0:
        text += '}' * brace_diff
    if bracket_diff > 0:
        text += ']' * bracket_diff
    return text


def _close_unterminated_strings(text: str) -> str:
    """Ensure the final JSON string closes any dangling double quotes."""
    result: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        result.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
    if in_string:
        result.append('"')
    return ''.join(result)


def _fix_mismatched_closers(text: str) -> str:
    """Replace stray closing braces/brackets with the expected counterpart."""
    result: List[str] = []
    stack: List[str] = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            result.append(ch)
            continue

        if ch in '{[':
            stack.append(ch)
            result.append(ch)
            continue

        if ch in '}]':
            if stack:
                opener = stack.pop()
                expected = '}' if opener == '{' else ']'
                result.append(expected if ch != expected else ch)
            else:
                result.append(ch)
            continue

        result.append(ch)

    return ''.join(result)


def _find_json_payload(text: str) -> str:
    """Extract the first JSON-like block from a mixed response."""
    first_object = text.find('{')
    first_array = text.find('[')
    candidates = [idx for idx in (first_object, first_array) if idx != -1]
    if not candidates:
        return text

    start = min(candidates)
    opening = text[start]
    stack: List[str] = [opening]
    in_string = False
    escape = False

    closing_map = {'{': '}', '[': ']'}
    for idx in range(start + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch in '{[':
            stack.append(ch)
        elif ch in '}]' and stack:
            expected = closing_map.get(stack[-1])
            if ch == expected:
                stack.pop()
                if not stack:
                    return text[start: idx + 1]
            else:
                continue

    return text[start:]


def _recover_from_json_error(text: str, error: json.JSONDecodeError) -> str:
    """Best-effort truncation + balancing when json.loads pinpoints a fault."""
    cutoff = max(0, min(len(text), error.pos))
    repaired = text[:cutoff]
    if not repaired.strip():
        return text
    repaired = _remove_incomplete_tail(repaired)
    repaired = _trim_to_valid_json_suffix(repaired)
    repaired = _close_unterminated_strings(repaired)
    repaired = _balance_json_brackets(repaired)
    return repaired


def extract_json_from_response(response_str: str) -> Any:
    """
    Sanitizes a string-based LLM response to ensure it's valid JSON.

    Args:
        response_str (str): The raw response string from the LLM.

    Returns:
        Any: The parsed JSON object.

    Raises:
        json.JSONDecodeError: If the string cannot be parsed as JSON after sanitization.
    """
    response_str = _strip_code_fences(response_str or "")
    response_str = _normalize_unicode_quotes(response_str)

    # Remove leading markdown bullets or blockquote markers (e.g., "- {" or "> {")
    response_str = re.sub(r"^\s*[-*>]\s+(?=[\[{])", "", response_str)
    response_str = response_str.strip()

    if not response_str:
        raise json.JSONDecodeError("LLM response is empty; JSON payload missing", response_str, 0)

    candidate = _find_json_payload(response_str)
    candidate = _trim_to_valid_json_suffix(candidate)
    candidate = _close_unterminated_strings(candidate)
    candidate = _fix_mismatched_closers(candidate)
    candidate = _balance_json_brackets(candidate)

    last_error: Optional[json.JSONDecodeError] = None
    for attempt in range(2):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            last_error = err
            repaired = _recover_from_json_error(candidate, err)
            if repaired == candidate:
                break
            candidate = repaired

    if last_error:
        try:
            return ast.literal_eval(candidate)
        except Exception:
            raise last_error

    try:
        return ast.literal_eval(candidate)
    except Exception as exc:
        raise json.JSONDecodeError("Unable to parse JSON content", candidate, 0) from exc


# ==========================
# Prompt + context utilities
# ==========================

def estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def clamp_text(text: str, max_chars: int) -> str:
    """Clamp a text to max_chars, appending an ellipsis if truncated."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _compact_memory_records(records: List[Dict[str, Any]],
                            max_items: int = 3,
                            max_user_input: int = 300,
                            max_final_response: int = 600) -> List[Dict[str, Any]]:
    """Create a compact projection of memory records to avoid prompt explosions.

    Expects items shaped like CognitiveCycle.model_dump() but only extracts
    a safe subset of fields and clamps long strings.
    """
    compact: List[Dict[str, Any]] = []
    for rec in records[:max_items]:
        try:
            compact.append({
                "cycle_id": rec.get("cycle_id"),
                "timestamp": rec.get("timestamp"),
                "user_input": clamp_text(rec.get("user_input", ""), max_user_input),
                "final_response": clamp_text(rec.get("final_response", ""), max_final_response),
                # optional light metadata if present
                "topics": rec.get("metadata", {}).get("topics", []) if isinstance(rec.get("metadata"), dict) else [],
            })
        except Exception:
            # Best effort: if structure unexpected, include minimal info
            compact.append({
                "cycle_id": rec.get("cycle_id"),
                "user_input": clamp_text(str(rec), max_user_input)
            })
    return compact


def compact_agent_outputs(other_agent_outputs: List[Any],
                          per_agent_max_chars: int = 8000,
                          total_max_chars: int = 30000) -> str:
    """Return a compact, size-bounded string summarizing other agents' outputs.

    - Special-cases memory_agent to include only compacted retrieved_context summary.
    - Clamps each agent block and overall total to prevent prompt explosions.
    """
    blocks: List[str] = []
    total_len = 0
    for out in other_agent_outputs or []:
        agent_id = getattr(out, "agent_id", "unknown_agent")
        analysis = getattr(out, "analysis", {})

        block_str = ""
        try:
            if agent_id == "memory_agent":
                # Expect analysis to contain retrieved_context
                recs: List[Dict[str, Any]] = analysis.get("retrieved_context", []) if isinstance(analysis, dict) else []
                compact_recs = _compact_memory_records(recs)
                compact_payload = {
                    "relevance_score": analysis.get("relevance_score", 0.0),
                    "retrieved_context": compact_recs,
                    "source_memory_ids": analysis.get("source_memory_ids", [])[:3],
                    "note": "memory context compacted to prevent oversized prompts"
                }
                block_str = f"- {agent_id}: " + json.dumps(compact_payload, separators=(",", ":"), cls=UUIDEncoder)
            else:
                # Generic agents: dump analysis compactly
                block_str = f"- {agent_id}: " + json.dumps(analysis, separators=(",", ":"), cls=UUIDEncoder)
        except Exception as e:
            logger.warning(f"Failed to serialize analysis for {agent_id}: {e}")
            block_str = f"- {agent_id}: {{\"error\": \"unserializable analysis\"}}"

        # Per-agent clamp
        if len(block_str) > per_agent_max_chars:
            block_str = clamp_text(block_str, per_agent_max_chars) + " [truncated]"

        # Overall clamp
        if total_len + len(block_str) > total_max_chars:
            remaining = max(0, total_max_chars - total_len)
            if remaining > 0:
                blocks.append(clamp_text(block_str, remaining) + " [truncated]")
            blocks.append("[additional agent outputs omitted due to size]")
            break

        blocks.append(block_str)
        total_len += len(block_str)

    return "\n".join(blocks)