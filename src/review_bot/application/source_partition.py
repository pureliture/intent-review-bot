"""Independent-version source spans, fitted against an actual model request.

Syntax is only a boundary preference. No dependency discovery, semantic matching,
target execution, byte-to-token estimate, or source truncation is performed.
"""
from __future__ import annotations

import bisect
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache


class PartitionError(ValueError):
    pass


@dataclass(frozen=True)
class SourceOriginal:
    evidence_id: str
    text: str
    kind: str
    path: str = ""
    version: str = ""
    sha: str = ""

    @cached_property
    def raw(self) -> bytes:
        return self.text.encode("utf-8")

    @cached_property
    def digest(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


@dataclass(frozen=True)
class SourceSpan:
    unit_id: str
    evidence_id: str
    source_hash: str
    start_byte: int
    end_byte: int
    begin_line: int
    end_line: int
    text: str
    kind: str
    path: str
    version: str
    sha: str
    boundary: str


def span_identity(original: SourceOriginal, start: int, end: int) -> str:
    identity = f"{original.evidence_id}:{original.digest}:{start}:{end}".encode()
    return "U" + hashlib.sha256(identity).hexdigest()[:24]


def make_span(original: SourceOriginal, start: int, end: int, boundary: str) -> SourceSpan:
    raw = original.raw
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(raw):
        raise PartitionError("invalid_original_range")
    try:
        text = raw[start:end].decode("utf-8")
        raw[:start].decode("utf-8")
        raw[:end].decode("utf-8")
    except UnicodeError:
        raise PartitionError("invalid_unicode_boundary") from None
    return SourceSpan(span_identity(original, start, end), original.evidence_id,
        original.digest, start, end, raw.count(b"\n", 0, start) + 1,
        raw.count(b"\n", 0, max(start, end - 1)) + 1, text, original.kind,
        original.path, original.version, original.sha, boundary)


@lru_cache(maxsize=2)
def _language(name: str):
    from tree_sitter import Language
    if name == "python":
        import tree_sitter_python
        return Language(tree_sitter_python.language())
    import tree_sitter_java
    return Language(tree_sitter_java.language())


def syntax_boundaries(original: SourceOriginal) -> tuple[tuple[int, ...], str]:
    """Return pinned grammar boundaries, keeping fitting functions intact."""
    path = original.path.lower()
    name = "python" if path.endswith(".py") else "java" if path.endswith(".java") else None
    if name is None:
        return (), "unsupported_language"
    from tree_sitter import Parser
    raw = original.raw
    root = Parser(_language(name)).parse(raw).root_node
    if root.has_error:
        return (), "syntax_error"
    functions = {"function_definition"} if name == "python" else {
        "method_declaration", "constructor_declaration", "compact_constructor_declaration"}
    allowed = ({"function_definition", "decorated_definition", "class_definition", "import_statement",
        "import_from_statement", "expression_statement", "if_statement", "for_statement",
        "while_statement", "try_statement", "with_statement", "match_statement", "comment"}
        if name == "python" else {"class_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "annotation_type_declaration", "method_declaration", "constructor_declaration",
        "compact_constructor_declaration", "field_declaration", "constant_declaration", "import_declaration",
        "package_declaration", "line_comment", "block_comment", "static_initializer", "enum_constant"})
    ends = [index + 1 for index, byte in enumerate(raw) if byte == 10]
    if not ends or ends[-1] != len(raw):
        ends.append(len(raw))
    stack, protected, candidates = [root], [], set()
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        if node.type in functions:
            parent = node.parent
            owner = parent if name == "python" and parent and parent.type == "decorated_definition" else node
            protected.append((owner.start_byte, owner.end_byte))
        if node.type in allowed:
            before = raw.rfind(b"\n", 0, node.start_byte) + 1
            index = bisect.bisect_left(ends, node.end_byte)
            after = ends[index] if index < len(ends) else len(raw)
            candidates.update((before, after))
    return tuple(cut for cut in sorted(candidates) if cut > 0 and
        not any(start < cut < end for start, end in protected)), "valid"


def partition_original(original: SourceOriginal, fits: Callable[[SourceSpan], bool], *,
    before_step: Callable[[], None] | None = None,
) -> tuple[SourceSpan, ...]:
    """Fit whole source, syntax boundaries, then lines/Unicode boundaries.

    ``fits`` must render and measure the complete consumer request. Every accepted
    span is independently checked; tokenizer additivity is never assumed.
    """
    raw = original.raw
    if before_step:
        before_step()
    whole = make_span(original, 0, len(raw), "whole")
    if fits(whole):
        return (whole,)
    if not raw:
        raise PartitionError("fixed_request_exceeds_input_budget")
    cuts, parser_status = syntax_boundaries(original)
    lines = [index + 1 for index, byte in enumerate(raw) if byte == 10]
    if not lines or lines[-1] != len(raw):
        lines.append(len(raw))
    spans, start = [], 0
    while start < len(raw):
        if before_step:
            before_step()
        tried: dict[int, bool] = {}

        def accepted(end: int) -> bool:
            if end not in tried:
                if before_step:
                    before_step()
                tried[end] = fits(make_span(original, start, end, "probe"))
            return tried[end]

        def search(points: list[int]) -> int | None:
            # BPE lengths need not be perfectly monotonic. Binary search is only
            # an efficiency heuristic; acceptance always uses a measured request.
            lo, hi, best = 0, len(points) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if accepted(points[mid]):
                    best, lo = points[mid], mid + 1
                else:
                    hi = mid - 1
            return best

        syntax = [cut for cut in cuts if cut > start]
        end = search(syntax) if syntax else None
        reason = "syntax_boundary"
        if end is None:
            end = search([cut for cut in lines if cut > start])
            reason = "line_fallback:" + parser_status
        if end is None:
            # Long lines are split at valid character boundaries, not discarded.
            next_line = next((cut for cut in lines if cut > start), len(raw))
            offsets, position = [], start
            for character in raw[start:next_line].decode("utf-8"):
                position += len(character.encode("utf-8"))
                offsets.append(position)
            end = search(offsets)
            reason = "unicode_boundary_fallback"
        if end is None or end <= start:
            raise PartitionError("fixed_request_exceeds_input_budget")
        span = make_span(original, start, end, reason)
        if not fits(span):
            raise PartitionError("partition_request_changed")
        spans.append(span)
        start = end
    verify_partition(original, tuple(spans))
    return tuple(spans)


def verify_partition(original: SourceOriginal, spans: tuple[SourceSpan, ...]) -> None:
    raw, cursor, recovered = original.raw, 0, []
    for span in spans:
        if span.start_byte != cursor or span.evidence_id != original.evidence_id or span.source_hash != original.digest:
            raise PartitionError("partition_coverage_mismatch")
        canonical = make_span(original, span.start_byte, span.end_byte, span.boundary)
        if asdict(canonical) != asdict(span):
            raise PartitionError("partition_content_mismatch")
        recovered.append(span.text.encode("utf-8"))
        cursor = span.end_byte
    if cursor != len(raw) or b"".join(recovered) != raw:
        raise PartitionError("partition_coverage_mismatch")
