"""Deterministic diff/syntax scope over immutable, explicitly prepared originals.

References are conservative syntactic candidates within the frozen corpus, not
compiler-resolved dependencies. No repository discovery or source rewriting.
"""
from __future__ import annotations

from collections import defaultdict, deque
import bisect
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import sys

from review_bot.application.intent_context import intent_path_exclusion, safe_relative_path
from review_bot.application.source_partition import (
    PartitionError, SourceOriginal, _language, make_span, partition_original,
)

POLICY = "diff-symbols/v1"
MAX_DEPTH = 5
MAX_DEPENDENCY_SYMBOLS = 128
MAX_DEPENDENCY_FILES = 32
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_FULL_CONTEXT_SOURCE_BYTES = 16 * 1024

_JAVA_EXTERNAL_PREFIXES = (
    "java.", "javax.", "jakarta.", "kotlin.", "scala.", "org.springframework.",
    "org.junit.", "org.mockito.", "org.assertj.", "org.slf4j.", "lombok.",
    "com.fasterxml.", "io.swagger.", "io.micrometer.", "reactor.", "org.apache.",
)
_JAVA_BUILTIN_TYPES = frozenset({
    "String", "Object", "Integer", "Long", "Boolean", "Double", "Float", "Byte",
    "Short", "Character", "Void", "Number", "Class", "Enum", "Record", "Exception",
    "RuntimeException", "IllegalArgumentException", "IllegalStateException",
    "NullPointerException", "UnsupportedOperationException", "Throwable", "Error",
    "Override", "Deprecated", "SuppressWarnings", "FunctionalInterface",
    "List", "Map", "Set", "Collection", "Iterable", "Iterator", "Optional",
    "Stream", "Collectors", "Arrays", "Collections", "Objects", "UUID",
    "Instant", "Duration", "LocalDate", "LocalDateTime", "ZonedDateTime",
    "OffsetDateTime", "BigDecimal", "BigInteger", " Path", "File", "CompletableFuture",
    "Mono", "Flux", "Logger", "LoggerFactory", "Test", "BeforeEach", "AfterEach",
    "ParameterizedTest", "ValueSource", "MethodSource", "DisplayName", "Nested",
    "Mock", "InjectMocks", "Captor", "ExtendWith", "Autowired", "Component",
    "Service", "Repository", "Controller", "RestController", "Configuration",
    "Bean", "Value", "Transactional", "Slf4j", "Getter", "Setter", "Builder",
    "Data", "RequiredArgsConstructor", "AllArgsConstructor", "NoArgsConstructor",
})
_PYTHON_EXTERNAL_MODULES = frozenset({
    *getattr(sys, "stdlib_module_names", ()),
    "pytest", "pydantic", "fastapi", "starlette", "httpx", "yaml", "uvicorn",
    "tree_sitter", "tree_sitter_java", "tree_sitter_python", "requests",
    "sqlalchemy", "numpy", "pandas",
})


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
        **({"unresolved_references": scope["unresolved_references"]} if scope.get("unresolved_references") else {}),
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
    is_container: bool = False


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
        symbols.append(Symbol(original.evidence_id, text, owner.start_byte, end, frozenset(references), is_type))
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


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    parts = set(lower.split("/")[:-1])
    name = lower.rsplit("/", 1)[-1]
    return bool(parts & {"test", "tests", "__tests__"}) or name.startswith("test_") or name.endswith(("test.java", "tests.java", "_test.py"))


def _java_source_roots(path: str, package_name: str) -> list[str]:
    directory = path.rsplit("/", 1)[0] if "/" in path else ""
    pkg_path = package_name.replace(".", "/") if package_name else ""
    roots: list[str] = []
    if pkg_path and (directory == pkg_path or directory.endswith("/" + pkg_path)):
        prefix = directory[: -len(pkg_path)].rstrip("/")
        roots.append(prefix)
        if prefix.endswith("src/test/java"):
            main_root = (prefix[: -len("src/test/java")] + "src/main/java").strip("/")
            if main_root not in roots:
                roots.insert(0, main_root)
    else:
        for candidate in ("src/main/java", ""):
            if candidate not in roots:
                roots.append(candidate)
    return roots


def discover_one_hop_dependency_paths(
    changed_files,
    base_files,
    head_files,
    *,
    existing_paths=(),
    max_files: int = MAX_DEPENDENCY_FILES,
) -> tuple[str, ...]:
    """Resolve 1-hop syntactic repository file candidates from changed Java/Python files."""
    by_base = {f.path: f for f in base_files}
    by_head = {f.path: f for f in head_files}
    excluded = set(existing_paths) | {
        p for change in changed_files for p in (change.path, change.previous_path) if p
    }
    prod_referenced: list[str] = []
    prod_imported: list[str] = []
    test_candidates: list[str] = []

    def add_candidate(bucket: list[str], candidate: str) -> None:
        try:
            clean = safe_relative_path(candidate)
        except Exception:
            return
        if clean in excluded or intent_path_exclusion(clean):
            return
        if clean not in bucket:
            bucket.append(clean)

    for change in changed_files:
        old_path = change.previous_path or change.path
        base_file = by_base.get(old_path)
        head_file = by_head.get(change.path)
        target_file = head_file or base_file
        if target_file is None or not target_file.path.endswith((".java", ".py")):
            continue
        is_test = _is_test_path(target_file.path)
        orig = SourceOriginal("H", target_file.content, "source", target_file.path, "head", "0" * 40)
        syms, _, _ = _syntax(orig)
        referenced_names: set[str] = set()
        if base_file is not None and head_file is not None:
            _, head_edits = _changed_ranges(base_file.content, head_file.content)
            for start, end in head_edits:
                overlapping = [s for s in syms if s.start < end and start < s.end]
                leaves = [
                    s for s in overlapping
                    if not any(s.start <= t.start and t.end <= s.end and (s.start, s.end) != (t.start, t.end) for t in overlapping)
                ]
                for sym in leaves:
                    referenced_names.update(sym.references)
                hunk_text = head_file.content.encode("utf-8")[start:end].decode("utf-8", errors="ignore")
                referenced_names.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", hunk_text))
        else:
            for sym in syms:
                referenced_names.update(sym.references)

        primary_bucket = test_candidates if is_test else prod_referenced
        secondary_bucket = test_candidates if is_test else prod_imported

        if target_file.path.endswith(".java"):
            pkg_match = re.search(r"^\s*package\s+([\w.]+)\s*;", target_file.content, re.M)
            pkg_name = pkg_match.group(1) if pkg_match else ""
            roots = _java_source_roots(target_file.path, pkg_name)
            explicit_types: set[str] = set()
            wildcard_pkgs: list[str] = []
            for imp_match in re.finditer(r"^\s*import\s+(static\s+)?([\w.*]+)\s*;", target_file.content, re.M):
                is_static = bool(imp_match.group(1))
                target_imp = imp_match.group(2)
                if target_imp.startswith(_JAVA_EXTERNAL_PREFIXES):
                    continue
                if target_imp.endswith(".*"):
                    base_imp = target_imp[:-2]
                    if is_static and "." in base_imp and base_imp.rsplit(".", 1)[-1][:1].isupper():
                        pkg_part, cls_name = base_imp.rsplit(".", 1)
                        explicit_types.add(cls_name)
                        for root in roots:
                            rel = f"{pkg_part.replace('.', '/')}/{cls_name}.java"
                            add_candidate(primary_bucket if cls_name in referenced_names else secondary_bucket, f"{root}/{rel}" if root else rel)
                    else:
                        wildcard_pkgs.append(base_imp)
                    continue
                parts = target_imp.split(".")
                if is_static and len(parts) >= 2 and parts[-2][:1].isupper():
                    parts = parts[:-1]
                cls_name = parts[-1]
                if not cls_name[:1].isupper():
                    continue
                explicit_types.add(cls_name)
                rel_path = "/".join(parts) + ".java"
                bucket = primary_bucket if cls_name in referenced_names else secondary_bucket
                for root in roots:
                    add_candidate(bucket, f"{root}/{rel_path}" if root else rel_path)

            declared_names = {s.name for s in syms if s.name}
            type_refs = sorted(
                name for name in referenced_names
                if name[:1].isupper() and name not in _JAVA_BUILTIN_TYPES and name not in explicit_types and name not in declared_names
            )
            if pkg_name:
                pkg_rel = pkg_name.replace(".", "/")
                for type_name in type_refs:
                    for root in roots:
                        rel = f"{pkg_rel}/{type_name}.java"
                        add_candidate(primary_bucket, f"{root}/{rel}" if root else rel)
            for wild_pkg in wildcard_pkgs:
                wild_rel = wild_pkg.replace(".", "/")
                for type_name in type_refs:
                    for root in roots:
                        rel = f"{wild_rel}/{type_name}.java"
                        add_candidate(secondary_bucket, f"{root}/{rel}" if root else rel)
        elif target_file.path.endswith(".py"):
            dir_parts = target_file.path.split("/")[:-1]
            for from_match in re.finditer(r"^\s*from\s+(\.*[\w.]*)\s+import\s+([^\n#;]+)", target_file.content, re.M):
                mod_spec = from_match.group(1).strip()
                imported_clause = from_match.group(2).strip().strip("()")
                imported_items = [item.strip().split(" as ")[0].strip() for item in imported_clause.split(",") if item.strip()]
                if mod_spec.startswith("."):
                    dots = len(mod_spec) - len(mod_spec.lstrip("."))
                    rel_mod = mod_spec[dots:]
                    up = dots - 1
                    if up <= len(dir_parts):
                        base_parts = dir_parts[: len(dir_parts) - up]
                        mod_parts = [p for p in rel_mod.split(".") if p]
                        full_parts = base_parts + mod_parts
                        if full_parts:
                            prefix = "/".join(full_parts)
                            add_candidate(primary_bucket, f"{prefix}.py")
                            add_candidate(primary_bucket, f"{prefix}/__init__.py")
                        for item_name in imported_items:
                            if item_name and item_name != "*":
                                item_prefix = "/".join([*full_parts, item_name])
                                add_candidate(primary_bucket, f"{item_prefix}.py")
                else:
                    top_pkg = mod_spec.split(".", 1)[0]
                    if top_pkg in _PYTHON_EXTERNAL_MODULES:
                        continue
                    rel_mod = mod_spec.replace(".", "/")
                    roots = ["src", ""] if target_file.path.startswith(("src/", "tests/")) else [""]
                    for root in roots:
                        base_rel = f"{root}/{rel_mod}" if root else rel_mod
                        add_candidate(primary_bucket, f"{base_rel}.py")
                        add_candidate(primary_bucket, f"{base_rel}/__init__.py")
                        for item_name in imported_items:
                            if item_name and item_name != "*":
                                add_candidate(secondary_bucket, f"{base_rel}/{item_name}.py")
            for imp_match in re.finditer(r"^\s*import\s+([^\n#;]+)", target_file.content, re.M):
                for clause in imp_match.group(1).split(","):
                    mod_spec = clause.strip().split(" as ")[0].strip()
                    if not mod_spec or mod_spec.startswith("."):
                        continue
                    top_pkg = mod_spec.split(".", 1)[0]
                    if top_pkg in _PYTHON_EXTERNAL_MODULES:
                        continue
                    rel_mod = mod_spec.replace(".", "/")
                    roots = ["src", ""] if target_file.path.startswith(("src/", "tests/")) else [""]
                    for root in roots:
                        base_rel = f"{root}/{rel_mod}" if root else rel_mod
                        add_candidate(secondary_bucket, f"{base_rel}.py")
                        add_candidate(secondary_bucket, f"{base_rel}/__init__.py")

    ordered: list[str] = []
    for group in (prod_referenced, prod_imported, test_candidates):
        for candidate in group:
            if candidate not in ordered:
                ordered.append(candidate)
            if len(ordered) >= max_files * 2:
                return tuple(ordered)
    return tuple(ordered)


def build_source_scope(originals, changes):
    """Select all changed syntax units, then at most five reference hops.

    Changed units are mandatory. Dependency caps never discard them; exceeding a
    cap stops expansion and records the boundary. Unparsed text keeps exact diff
    hunks plus three context lines. Added/deleted files retain their full text.
    """
    sources = {o.evidence_id: o for o in originals}
    by_location = {(o.version, o.path): o for o in originals if o.kind in {"source", "test_source", "context_source"}}
    selected = {o.evidence_id: ([] if o.kind == "context_source" else [(0, len(o.raw))]) for o in originals}
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
    # Entire added files also seed dependency discovery (excluding context_source).
    for original in by_location.values():
        if original.kind != "context_source" and selected[original.evidence_id] == [(0, len(original.raw))]:
            seeds.update(symbols[original.evidence_id])
    queue = deque((symbol, 0) for symbol in sorted(seeds, key=lambda s: (s.evidence_id, s.start, s.end)))
    visited = set(seeds)
    dependency_count, files, boundaries = 0, set(), set()
    unresolved_by_symbol: dict[str, str] = {}
    while queue:
        symbol, depth = queue.popleft()
        if depth >= MAX_DEPTH:
            continue
        for name in sorted(symbol.references - {symbol.name}):
            candidates = index.get((sources[symbol.evidence_id].version, name), [])
            local = [s for s in candidates if s.evidence_id == symbol.evidence_id]
            candidates = local or candidates
            if len(candidates) != 1:
                if depth == 0 and name[:1].isupper() and name not in _JAVA_BUILTIN_TYPES:
                    reason = "ambiguous_candidates" if len(candidates) > 1 else "outside_fixed_corpus"
                    unresolved_by_symbol.setdefault(name, reason)
                continue  # Ambiguous/unavailable names are not resolved by spelling alone.
            target = candidates[0]
            if target in visited:
                continue
            path = sources[target.evidence_id].path
            if dependency_count >= MAX_DEPENDENCY_SYMBOLS or path not in files and len(files) >= MAX_DEPENDENCY_FILES:
                boundaries.add("dependency_expansion_limit")
                if depth == 0:
                    unresolved_by_symbol.setdefault(name, "dependency_expansion_limit")
                continue
            visited.add(target)
            dependency_count += 1
            files.add(path)
            target_orig = sources[target.evidence_id]
            if target_orig.kind == "context_source" and target.is_container and len(target_orig.raw) <= MAX_FULL_CONTEXT_SOURCE_BYTES:
                selected[target.evidence_id].append((0, len(target_orig.raw)))
            else:
                selected[target.evidence_id].append((target.start, target.end))
                selected[target.evidence_id].extend(contexts[target.evidence_id])
            queue.append((target, depth + 1))
    for original in by_location.values():
        if original.kind == "context_source" and not selected[original.evidence_id]:
            if len(original.raw) <= MAX_FULL_CONTEXT_SOURCE_BYTES or not symbols[original.evidence_id]:
                selected[original.evidence_id] = [(0, len(original.raw))]
            else:
                selected[original.evidence_id] = list(contexts[original.evidence_id]) + [
                    (s.start, s.end) for s in symbols[original.evidence_id][:16]
                ]
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
    unresolved_list = [
        {"symbol": sym, "reason": reason}
        for sym, reason in sorted(unresolved_by_symbol.items())[:64]
    ]
    return {"policy": POLICY, "max_depth": MAX_DEPTH,
        "max_dependency_symbols": MAX_DEPENDENCY_SYMBOLS, "max_dependency_files": MAX_DEPENDENCY_FILES,
        "resolution": "syntactic_candidates_in_fixed_corpus", "boundaries": sorted(boundaries),
        **({"unresolved_references": unresolved_list} if unresolved_list else {}),
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
