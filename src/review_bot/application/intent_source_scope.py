"""Deterministic diff/syntax scope over immutable, explicitly prepared originals.

References are conservative syntactic candidates within the frozen corpus, not
compiler-resolved dependencies. No repository discovery or source rewriting.
"""
from __future__ import annotations

from collections import defaultdict, deque
import bisect
from dataclasses import dataclass
from difflib import SequenceMatcher

from review_bot.application.source_partition import (
    PartitionError, SourceOriginal, _language, make_span, partition_original,
)

POLICY = "diff-symbols/v1"
MAX_DEPTH = 5
MAX_DEPENDENCY_SYMBOLS = 128
MAX_DEPENDENCY_FILES = 32
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def scope_notice(meta, identifiers):
    scope = meta.get("source_scope")
    if not scope:
        return {}
    return {"declared_source_scope": {"policy": scope["policy"],
        "meaning": "Review changed declarations and bounded syntactic reference candidates in the fixed corpus. "
            "Only available_ranges are declared for source delivery, including rechecks. Original file hashes and "
            "offsets remain intact. Do not claim to have read unselected file bytes. Report a genuinely necessary "
            "unavailable dependency explicitly; unexecuted tests alone do not prevent static review.",
        "boundaries": scope["boundaries"],
        "available_ranges": {key: ranges for key, ranges in scope["ranges"].items() if key in identifiers}}}


def merge_ranges(ranges):
    result = []
    for start, end in sorted({tuple(value) for value in ranges}):
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


@dataclass(frozen=True)
class Symbol:
    evidence_id: str
    name: str
    start: int
    end: int
    references: frozenset[str]


def _syntax(original):
    name = "python" if original.path.endswith(".py") else "java" if original.path.endswith(".java") else None
    if name is None:
        return [], [], "unsupported_language"
    from tree_sitter import Parser
    root = Parser(_language(name)).parse(original.raw).root_node
    if root.has_error:
        return [], [], "syntax_error"
    declarations = ({"function_definition", "class_definition"} if name == "python" else {
        "method_declaration", "constructor_declaration", "compact_constructor_declaration",
        "class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
        "field_declaration", "constant_declaration", "static_initializer"})
    context_types = {"import_statement", "import_from_statement", "package_declaration", "import_declaration"}
    symbols, context, stack = [], [], [root]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        if node.type in context_types:
            context.append((node.start_byte, node.end_byte))
        if node.type not in declarations:
            continue
        owner = node.parent if name == "python" and node.parent and node.parent.type == "decorated_definition" else node
        label = node.child_by_field_name("name")
        if label is None and node.type in {"field_declaration", "constant_declaration"}:
            declarator = next((c for c in node.named_children if c.type == "variable_declarator"), None)
            label = declarator.child_by_field_name("name") if declarator else None
        body = node.child_by_field_name("body")
        is_type = node.type in {"class_definition", "class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}
        end = body.start_byte + (name == "java") if is_type and body is not None else owner.end_byte
        if is_type:
            context.append((owner.start_byte, end))
        references, pending = set(), [c for c in node.named_children if not is_type or c != body]
        while pending:
            child = pending.pop()
            pending.extend(child.named_children)
            if child.type in {"identifier", "type_identifier"}:
                references.add(original.raw[child.start_byte:child.end_byte].decode())
        text = original.raw[label.start_byte:label.end_byte].decode() if label else ""
        symbols.append(Symbol(original.evidence_id, text, owner.start_byte, end, frozenset(references)))
    return symbols, context, "valid"


def _changed_ranges(before, after):
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    def offsets(lines):
        result = [0]
        for line in lines:
            result.append(result[-1] + len(line.encode()))
        return result
    offsets_by_side = [offsets(old), offsets(new)]
    ranges = [[], []]
    for tag, i, j, k, l in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for side, start, end, lines in ((0, i, j, old), (1, k, l, new)):
            positions = offsets_by_side[side]
            if start != end:
                ranges[side].append((positions[start], positions[end]))
            elif lines:
                anchor = min(start, len(lines) - 1)
                ranges[side].append((positions[anchor], positions[anchor + 1]))
    return ranges


def build_source_scope(originals, changes):
    """Select all changed syntax units, then at most five reference hops.

    Changed units are mandatory. Dependency caps never discard them; exceeding a
    cap stops expansion and records the boundary. Unparsed text keeps exact diff
    hunks plus three context lines. Added/deleted files retain their full text.
    """
    sources = {o.evidence_id: o for o in originals}
    by_location = {(o.version, o.path): o for o in originals if o.kind in {"source", "test_source"}}
    selected = {o.evidence_id: [(0, len(o.raw))] for o in originals}
    symbols, contexts, statuses = {}, {}, {}
    for original in by_location.values():
        if len(original.raw) > MAX_SOURCE_BYTES:
            symbols[original.evidence_id], contexts[original.evidence_id], statuses[original.evidence_id] = [], [], "source_byte_limit"
        else:
            symbols[original.evidence_id], contexts[original.evidence_id], statuses[original.evidence_id] = _syntax(original)
    seeds, changed_selection = set(), defaultdict(list)
    for change in changes:
        base = by_location.get(("base", change.previous_path or change.path))
        head = by_location.get(("head", change.path))
        if base is None or head is None:
            continue
        ranges = _changed_ranges(base.text, head.text)
        for original, edits in zip((base, head), ranges):
            if not edits:
                continue  # Identical explicit context remains available in full.
            identifier = original.evidence_id
            chosen = []
            lines = original.text.splitlines(keepends=True)
            offsets = [0]
            for line in lines:
                offsets.append(offsets[-1] + len(line.encode()))
            for start, end in edits:
                # Every edited line must be represented, including edits outside declarations.
                overlapping = [s for s in symbols[identifier] if s.start < end and start < s.end]
                leaves = [s for s in overlapping if not any(s.start <= t.start and t.end <= s.end
                    and (s.start, s.end) != (t.start, t.end) for t in overlapping)]
                seeds.update(leaves)
                chosen.extend((s.start, s.end) for s in leaves)
                # Keep a literal hunk even when AST declarations do not cover it.
                begin = max(0, bisect.bisect_right(offsets, start) - 1 - 3)
                finish = min(len(lines), bisect.bisect_left(offsets, end) + 3)
                chosen.append((offsets[begin], offsets[finish]))
            changed_selection[identifier].extend(chosen + contexts[identifier])
    selected.update(changed_selection)
    index = defaultdict(list)
    for identifier, entries in symbols.items():
        for symbol in entries:
            if symbol.name:
                index[(sources[identifier].version, symbol.name)].append(symbol)
    # Entire added files also seed dependency discovery.
    for original in by_location.values():
        if selected[original.evidence_id] == [(0, len(original.raw))]:
            seeds.update(symbols[original.evidence_id])
    queue = deque((symbol, 0) for symbol in sorted(seeds, key=lambda s: (s.evidence_id, s.start, s.end)))
    visited = set(seeds)
    dependency_count, files, boundaries = 0, set(), set()
    while queue:
        symbol, depth = queue.popleft()
        if depth >= MAX_DEPTH:
            continue
        for name in sorted(symbol.references - {symbol.name}):
            candidates = index.get((sources[symbol.evidence_id].version, name), [])
            local = [s for s in candidates if s.evidence_id == symbol.evidence_id]
            candidates = local or candidates
            if len(candidates) != 1:
                continue  # Ambiguous/unavailable names are not resolved by spelling alone.
            target = candidates[0]
            if target in visited:
                continue
            path = sources[target.evidence_id].path
            if dependency_count >= MAX_DEPENDENCY_SYMBOLS or path not in files and len(files) >= MAX_DEPENDENCY_FILES:
                boundaries.add("dependency_expansion_limit")
                continue
            visited.add(target)
            dependency_count += 1
            files.add(path)
            selected[target.evidence_id].append((target.start, target.end))
            selected[target.evidence_id].extend(contexts[target.evidence_id])
            queue.append((target, depth + 1))
    def delivery_ranges(ranges):
        # Avoid a unit for every import/annotation. Added intervening bytes are
        # explicit members of the selected scope and retain original offsets.
        result = []
        for start, end in merge_ranges(ranges):
            if result and start - result[-1][1] <= 128:
                result[-1][1] = end
            else:
                result.append([start, end])
        return result
    return {"policy": POLICY, "max_depth": MAX_DEPTH,
        "max_dependency_symbols": MAX_DEPENDENCY_SYMBOLS, "max_dependency_files": MAX_DEPENDENCY_FILES,
        "resolution": "syntactic_candidates_in_fixed_corpus", "boundaries": sorted(boundaries),
        "parser_status": dict(sorted(statuses.items())),
        "ranges": {key: delivery_ranges(value) for key, value in sorted(selected.items())}}


def scoped_ranges(original, scope):
    return scope["ranges"][original.evidence_id] if scope else [[0, len(original.raw)]]


def partition_scoped_original(original, fits, scope, *, before_step=None):
    result = []
    for start, end in scoped_ranges(original, scope):
        local = SourceOriginal(original.evidence_id, original.raw[start:end].decode(), original.kind,
            original.path, original.version, original.sha)
        def absolute(span):
            return make_span(original, start + span.start_byte, start + span.end_byte, span.boundary)
        parts = partition_original(local, lambda part: fits(absolute(part)), before_step=before_step)
        result.extend(absolute(part) for part in parts)
    return tuple(result)


def verify_scoped_partition(original, spans, scope):
    expected = scoped_ranges(original, scope)
    if merge_ranges((s.start_byte, s.end_byte) for s in spans) != merge_ranges(expected):
        raise PartitionError("partition_scope_coverage_mismatch")
    cursor = None
    for span in spans:
        if cursor is not None and span.start_byte < cursor:
            raise PartitionError("partition_scope_overlap")
        if span != make_span(original, span.start_byte, span.end_byte, span.boundary):
            raise PartitionError("partition_content_mismatch")
        cursor = span.end_byte
