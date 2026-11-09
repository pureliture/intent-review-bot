"""Fail-closed byte-upper-bound input gate shared by all model adapters.

Every token covers at least one UTF-8 byte, so the byte length of a
admitted input representation bounds the tokens our own prompt can occupy.
Adapters using a full wire body are conservative; llama.cpp excludes its
response_format output grammar, which is not prompt content. A fixed framing reserve covers provider/server-side template
framing we cannot see. Anything beyond the reserve is caught post-send
by the response usage check and fails closed.

No tokenization, no estimation, no measurement calls.
"""
from __future__ import annotations

from review_bot.application.architecture_review import PermanentModelError

# stealth/crouton-alpha: uniqueItems returned "Provider returned error"; 512 (2/2) and 768 (1/2) ended finish_reason=length/content=None; 1024, 1536, 2048, 4096 completed (~/eval/select_payload.json).
ADMISSION_OUTPUT_FLOOR_TOKENS = 2048
# Closed choices are tiny requests (<= 64 output tokens) answered by
# mandatory-reasoning models. Reasoning alone burns hundreds of tokens, so
# choice wires get a larger floor. Content stays schema-bound to one small
# object; usage is still validated post-send. Long-term, closed choices
# belong on a decisions API, not a generative model.
_CHOICE_REQUEST_CEILING_TOKENS = 64
CHOICE_WIRE_FLOOR_TOKENS = 2048

# Fixed conservative cover for server-side framing (chat templates and
# request envelopes) that is not part of our wire bytes. A post-send usage
# check fails closed if real framing ever exceeds it.
FRAMING_RESERVE_TOKENS = 8192


def effective_output_reservation(max_output_tokens: int, provider: str) -> int:
    if provider != "openrouter":
        return max_output_tokens
    floor = ADMISSION_OUTPUT_FLOOR_TOKENS
    if max_output_tokens <= _CHOICE_REQUEST_CEILING_TOKENS:
        floor = max(floor, CHOICE_WIRE_FLOOR_TOKENS)
    return max(max_output_tokens, floor)


def precheck_byte_budget(wire_len: int, max_output_tokens: int, *, app_input_tokens: int,
        app_output_tokens: int, model_context_tokens: int, role: str,
        framing_reserve_tokens: int = FRAMING_RESERVE_TOKENS) -> None:
    """Fail-closed pre-send gate. Raises PermanentModelError on any exceed.

    ``wire_len`` is UTF-8 bytes of the adapter's admitted input representation
    (the full wire body, or llama.cpp's body without output-grammar configuration).
    Since every token spans >= 1 byte, ``wire_len +
    framing_reserve_tokens`` is a proven input-token upper bound, never
    an estimate.
    """
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise PermanentModelError(f"Invalid {role} output reservation")
    if type(framing_reserve_tokens) is not int or framing_reserve_tokens < 0:
        raise PermanentModelError(f"Invalid {role} framing reserve")
    total_cap = min(app_input_tokens + app_output_tokens, model_context_tokens)
    if wire_len + framing_reserve_tokens + max_output_tokens > total_cap:
        raise PermanentModelError(f"{role} input exceeds byte-upper-bound budget; truncation is forbidden")
    if wire_len + framing_reserve_tokens > app_input_tokens:
        raise PermanentModelError(f"{role} input exceeds byte-upper-bound budget; truncation is forbidden")
