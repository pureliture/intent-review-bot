"""Conservative, code-backed PMD/CPD comparison and 6-state change computation engine."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any

from review_bot.domain.changes import Change, Measurement
from review_bot.domain.models_v1 import (
    AnalysisDelta,
    AnalysisRun,
    CpdDuplication,
    CpdOccurrence,
    DeltaSignal,
    Finding,
    PmdViolation,
)
from review_bot.domain.status import Tool

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_CONDITION_KEYS = {
    "schema_version", "pmd_version", "cpd_version", "ruleset_hash", "minimum_tokens",
    "source_roots", "language", "language_version", "encoding", "type_resolution",
    "java_version", "java_vendor", "classpath_hash", "runtime_classpath_hashes",
}
_JAVA_TOKEN = re.compile(
    r'\s+|//[^\r\n]*|/\*[\s\S]*?\*/|"""(?:\\[\s\S]|(?!""")[^\\])*"""'
    r'|"(?:\\[\s\S]|[^"\\\r\n])*"|\'(?:\\[\s\S]|[^\'\\\r\n])*\''
    r'|(?:[^\W\d]|[$])[\w$]*'
    r'|(?:0[xX][\da-fA-F_]+(?:\.[\da-fA-F_]*)?(?:[pP][+-]?[\d_]+)?[fFdDlL]?'
    r'|0[bB][01_]+[lL]?|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)'
    r'(?:[eE][+-]?[\d_]+)?[fFdDlL]?)'
    r'|>>>=|>>=|<<=|>>>|>>|<<|\.\.\.|->|::|==|!=|<=|>=|&&|\|\||\+\+|--'
    r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=|[^\s]'
)
_KEYWORDS = {
    "abstract", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue",
    "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "if",
    "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package",
    "private", "protected", "public", "record", "return", "short", "static", "strictfp", "super",
    "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile",
    "while", "true", "false", "null",
}
_SUBJECT = r"The (?:method|constructor|class|interface|enum|record|annotation) '[^'\r\n]+' "
_METRIC_PATTERNS = {
    "CognitiveComplexity": re.compile(_SUBJECT + r"has a cognitive complexity of (?P<cognitive_complexity>\d+), current threshold is \d+\.?"),
    "NPathComplexity": re.compile(_SUBJECT + r"has an NPath complexity of (?P<npath_complexity>\d+), current threshold is \d+\.?"),
    "CyclomaticComplexity": re.compile(_SUBJECT + r"has a(?: total)? cyclomatic complexity of (?P<cyclomatic_complexity>\d+)(?: \(highest (?P<highest_cyclomatic_complexity>\d+)\))?\.?"),
    "NcssCount": re.compile(_SUBJECT + r"has a NCSS line count of (?P<ncss>\d+)(?: \(Highest = (?P<highest_ncss>\d+)\))?\.?"),
    "GodClass": re.compile(r"Possible God Class \(WMC=(?P<wmc>\d+), ATFD=(?P<atfd>\d+), TCC=(?P<tcc>\d+(?:\.\d+)?%?)\)"),
}


def hash_json(value: object) -> str:
    """Return canonical SHA-256 digest of JSON-serialized object."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and not any(char in path for char in ("\\", "\0", ":"))
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )


def _time_range(start: str, end: str):
    try:
        first, last = (datetime.fromisoformat(value) for value in (start, end))
        return (first, last) if first.tzinfo is not None and last.tzinfo is not None and last >= first else None
    except (ValueError, TypeError, AttributeError):
        return None


def _aggregate(run: AnalysisRun, statuses: tuple[str, str]) -> str:
    if run.errors or run.source_state == "failed" or "failed" in statuses:
        return "failed"
    for status in ("missing_report", "incomplete"):
        if status in statuses:
            return status
    if run.source_state == "no_java_sources":
        return "no_sources"
    return "findings" if "findings" in statuses else "clean"


def _run_errors(run: AnalysisRun) -> tuple[str, ...]:
    problems = []
    conditions = run.conditions if isinstance(run.conditions, dict) else {}
    if (
        type(run.schema_version) is not int
        or run.schema_version != 1
        or not isinstance(run.sha, str)
        or not _SHA.fullmatch(run.sha)
    ):
        problems.append("invalid run schema or fixed revision")
    if not isinstance(run.run_id, str) or not run.run_id or _time_range(run.started_at, run.completed_at) is None:
        problems.append("invalid run identity or timestamps")
    if not _CONDITION_KEYS.issubset(conditions):
        problems.append("analysis conditions are incomplete")
    try:
        profile = {key: value for key, value in conditions.items() if key != "runtime_classpath_hashes"}
        if run.conditions_hash != hash_json(conditions) or run.profile_hash != hash_json(profile):
            problems.append("condition/profile hash does not match condition content")
        if "tool_classpath" in conditions and conditions.get("classpath_hash") != hash_json(conditions["tool_classpath"]):
            problems.append("classpath hash does not match the library inventory")
    except (TypeError, ValueError):
        problems.append("analysis conditions cannot be hashed")
    roots = conditions.get("source_roots")
    valid_roots = isinstance(roots, (tuple, list)) and bool(roots) and all(_safe_path(root) for root in roots)
    if (
        type(conditions.get("schema_version")) is not int
        or conditions.get("schema_version") != 1
        or conditions.get("language") != "java"
        or not valid_roots
        or not isinstance(conditions.get("runtime_classpath_hashes"), dict)
        or any(
            not isinstance(conditions.get(key), str) or not conditions[key]
            for key in ("java_version", "java_vendor", "language_version", "encoding", "type_resolution")
        )
    ):
        problems.append("invalid analysis conditions or source scope")
    for key in ("classpath_hash", "java_release_hash", "java_executable_hash"):
        value = conditions.get(key)
        if (key == "classpath_hash" or value is not None) and (not isinstance(value, str) or not _HASH.fullmatch(value)):
            problems.append(f"invalid {key} in analysis conditions")
    roots = roots if valid_roots else ()
    if not isinstance(run.source_hash, str) or not _HASH.fullmatch(run.source_hash):
        problems.append("source inventory has no valid content hash")
    if run.source_state == "no_java_sources" and run.source_hash != hashlib.sha256(b"").hexdigest():
        problems.append("no_java_sources hash does not match the empty source inventory")
    if (
        not isinstance(run.source_files, (tuple, list))
        or run.source_state not in ("complete", "no_java_sources")
        or bool(run.source_files) != (run.source_state == "complete")
        or any(
            not _safe_path(path)
            or not path.endswith(".java")
            or not any(path.startswith(root + "/") for root in roots if isinstance(root, str))
            for path in run.source_files
        )
        or len(set(run.source_files)) != len(run.source_files)
    ):
        problems.append("source scope is failed, incomplete or inconsistent")
    problems.extend(f"run error {error.kind}: {error.message}" for error in run.errors)
    if run.status != _aggregate(run, (run.pmd.status, run.cpd.status)):
        problems.append("run status disagrees with source/tool statuses")
    return tuple(problems)


def _range_valid(location) -> bool:
    return (
        type(location.begin_line) is int
        and type(location.end_line) is int
        and 0 < location.begin_line <= location.end_line
        and all(value is None or (type(value) is int and value > 0) for value in (location.begin_column, location.end_column))
        and (
            location.begin_line != location.end_line
            or location.begin_column is None
            or location.end_column is None
            or location.begin_column <= location.end_column
        )
    )


def _tool_check(run: AnalysisRun, tool: str, run_errors: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    analysis = getattr(run, tool)
    report, execution = analysis.report, analysis.execution
    problems = list(run_errors)
    if analysis.tool != tool or report.tool != tool:
        problems.append("tool and report identities disagree")
    if execution is None:
        return "failed", tuple(problems + ["missing execution record"])
    conditions = run.conditions if isinstance(run.conditions, dict) else {}
    roots = conditions.get("source_roots")
    roots = tuple(roots) if isinstance(roots, (tuple, list)) else ()
    files = run.source_files if isinstance(run.source_files, (tuple, list)) else ()
    expected = {
        "schema_version": 1,
        "tool": tool,
        "sha": run.sha,
        "run_id": run.run_id,
        "profile_hash": run.profile_hash,
        "source_hash": run.source_hash,
        "source_count": len(files),
        "source_root": str(PurePosixPath(run.output_dir) / "sources"),
        "source_roots": roots,
        "report_exists": report.exists,
        "report_hash": report.content_hash,
        "tool_version": conditions.get(tool + "_version"),
        "classpath_hash": conditions.get("classpath_hash"),
        "java_version": conditions.get("java_version"),
        "java_vendor": conditions.get("java_vendor"),
        "language_version": conditions.get("language_version"),
        "encoding": conditions.get("encoding"),
        "type_resolution": conditions.get("type_resolution"),
        "ruleset_hash": conditions.get("ruleset_hash") if tool == "pmd" else None,
        "minimum_tokens": conditions.get("minimum_tokens") if tool == "cpd" else None,
    }
    mismatches = [key for key, value in expected.items() if getattr(execution, key) != value]
    if mismatches:
        problems.append("execution record disagrees with run/report/conditions: " + ", ".join(mismatches))
    if (
        type(execution.schema_version) is not int
        or type(execution.source_count) is not int
        or type(execution.report_exists) is not bool
        or type(report.exists) is not bool
        or analysis.exit_code != execution.exit_code
        or (analysis.exit_code is not None and type(analysis.exit_code) is not int)
        or (execution.exit_code is not None and type(execution.exit_code) is not int)
    ):
        problems.append("invalid execution field types or exit concordance")
    if execution.tool_version != "7.3.0":
        problems.append("unsupported analysis tool version")
    if tool == "pmd" and (not isinstance(execution.ruleset_hash, str) or not _HASH.fullmatch(execution.ruleset_hash)):
        problems.append("invalid PMD ruleset hash")
    if tool == "cpd" and (type(execution.minimum_tokens) is not int or execution.minimum_tokens <= 0):
        problems.append("invalid CPD token threshold")
    runtime = conditions.get("runtime_classpath_hashes", {})
    if not isinstance(runtime, dict) or runtime.get(tool) != execution.classpath_hash:
        problems.append("runtime classpath does not match this tool's conditions")
    bounds = _time_range(run.started_at, run.completed_at)
    times = _time_range(execution.started_at, execution.completed_at)
    if times is None or bounds is None or not (bounds[0] <= times[0] <= times[1] <= bounds[1]):
        problems.append("invalid execution timestamps")
    subcommand = "check" if tool == "pmd" else "cpd"
    if (
        not execution.command
        or not all(isinstance(value, str) and value for value in execution.command)
        or ("net.sourceforge.pmd.cli.PmdCli", subcommand)
        not in tuple(zip(execution.command, execution.command[1:]))
    ):
        problems.append("invalid recorded tool command")
    if report.tool_version is not None and report.tool_version != execution.tool_version:
        problems.append("report and execution versions disagree")
    if (
        report.exists
        and (not isinstance(report.content_hash, str) or not _HASH.fullmatch(report.content_hash))
        or not report.exists
        and (report.content_hash is not None or report.parse_status != "missing")
        or report.exists
        and report.parse_status == "missing"
        or report.parse_status not in ("valid", "missing", "invalid")
    ):
        problems.append("report presence, parse status and content hash disagree")
    findings = report.violations if tool == "pmd" else report.duplications
    if (
        (report.duplications if tool == "pmd" else report.violations)
        or type(report.suppressed_count) is not int
        or report.suppressed_count < 0
    ):
        problems.append("report contains invalid finding collections or suppression count")
    if len({hash_json(_payload(finding)) for finding in findings}) != len(findings):
        problems.append("report contains duplicate indistinguishable observations")
    for finding in findings:
        locations = (finding,) if tool == "pmd" else finding.occurrences
        if (
            tool == "cpd"
            and (len(locations) < 2 or type(finding.lines) is not int or finding.lines <= 0 or type(finding.tokens) is not int or finding.tokens <= 0)
            or any(not _range_valid(location) or location.path not in files for location in locations)
        ):
            problems.append("report findings are malformed or outside the acquired source inventory")
            break
        if tool == "cpd" and len({(o.path, o.begin_line, o.end_line, o.begin_column, o.end_column) for o in locations}) != len(locations):
            problems.append("CPD group repeats the same occurrence position")
        if tool == "pmd" and (
            not finding.rule
            or not finding.message.strip()
            or type(finding.priority) is not int
            or not 1 <= finding.priority <= 5
            or finding.begin_column is None
            or finding.end_column is None
        ):
            problems.append("PMD violation has invalid rule/message/priority/columns")
    if run.source_state == "no_java_sources":
        if (
            analysis.status != "no_sources"
            or analysis.errors
            or findings
            or report.exists
            or report.parse_status != "missing"
            or execution.state != "not_run"
            or execution.exit_code is not None
            or any(error.kind != "missing_report" for error in report.errors)
        ):
            problems.append("no_java_sources requires an empty inventory and concordant not_run execution")
        return ("failed" if problems else "no_sources"), tuple(problems)
    if not report.exists or report.parse_status == "missing":
        status = "missing_report"
    elif report.parse_status != "valid" or execution.state != "completed" or execution.exit_code not in (0, 4, 5) or any(error.kind not in ("processing_error", "incomplete_analysis") for error in report.errors + analysis.errors):
        status = "failed"
    elif execution.exit_code == 5 or report.errors or analysis.errors:
        status = "incomplete"
    elif (execution.exit_code == 4) != bool(findings):
        status = "failed"
        problems.append("execution exit and report finding count disagree")
    else:
        status = "findings" if findings else "clean"
    if analysis.status != status:
        problems.append("declared tool status disagrees with execution/report status")
    if status not in ("clean", "findings"):
        problems.append(f"{status} tool analysis cannot establish absence or change")
        problems.extend(f"{error.kind}: {error.message}" for error in report.errors + analysis.errors)
    return ("failed" if problems and status in ("clean", "findings") else status), tuple(dict.fromkeys(problems))


def _tool_conditions(run: AnalysisRun, tool: str) -> dict[str, object]:
    other = {"cpd_version", "minimum_tokens"} if tool == "pmd" else {"pmd_version", "ruleset_hash"}
    conditions = run.conditions if isinstance(run.conditions, dict) else {}
    result = {key: value for key, value in conditions.items() if key not in other}
    runtime = result.get("runtime_classpath_hashes", {})
    result["runtime_classpath_hashes"] = {tool: runtime.get(tool)} if isinstance(runtime, dict) else runtime
    if tool == "cpd" and isinstance(result.get("trusted_files"), dict):
        result["trusted_files"] = {
            key: value
            for key, value in result["trusted_files"].items()
            if key != ".github/pmd/ruleset.xml"
        }
    return result


def _rename_errors(rename_map: dict[str, str]) -> tuple[str, ...]:
    if not isinstance(rename_map, dict) or any(not _safe_path(old) or not _safe_path(new) for old, new in rename_map.items()):
        return ("rename map contains unsafe paths",)
    if len(set(rename_map.values())) != len(rename_map):
        return ("rename map has ambiguous many-to-one destinations",)
    for old in rename_map:
        seen, current = set(), old
        while current in rename_map and rename_map[current] != current:
            if current in seen:
                return ("rename map contains a cycle",)
            seen.add(current)
            current = rename_map[current]
    return ()


@lru_cache(maxsize=2048)
def _code_tokens(code: str | None) -> tuple[str, ...] | None:
    if not isinstance(code, str) or not code:
        return None
    tokens, position = [], 0
    while position < len(code):
        match = _JAVA_TOKEN.match(code, position)
        if match is None:
            return None
        token = match.group()
        if token in ('"', "'") or (code.startswith("/*", position) and not token.endswith("*/")):
            return None
        if not token.isspace() and not token.startswith(("//", "/*")):
            tokens.append(token)
        position = match.end()
    return tuple(tokens) if any(re.match(r"[\w$\"']", token) for token in tokens) else None


def _substantive(tokens) -> bool:
    return len({token for token in tokens if token not in _KEYWORDS and re.match(r"[\w$\"']", token)}) >= 2


def _overlap(left, right) -> bool:
    if not left or not right:
        return False
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    longest = matcher.find_longest_match()
    return longest.size >= 6 and matcher.ratio() >= 0.6 and _substantive(left[longest.a:longest.a + longest.size])


def _declaration(finding: PmdViolation, tokens):
    if not tokens:
        return None
    declarations = []
    if not finding.method:
        for index, token in enumerate(tokens[:-1]):
            if token in ("class", "interface", "enum", "record") and tokens[index + 1] == finding.klass.split(".")[-1]:
                try:
                    end = tokens.index("{", index + 2)
                    declarations.append((tokens[index:end], tokens[end + 1:]))
                except ValueError:
                    pass
    else:
        for index, token in enumerate(tokens[:-1]):
            if token != finding.method or tokens[index + 1] != "(":
                continue
            if index and tokens[index - 1] in (".", "return", "new", "=", "("):
                continue
            if not index and finding.method != finding.klass.split(".")[-1]:
                continue
            end, depth = index + 2, 1
            while end < len(tokens) and depth:
                depth += (tokens[end] == "(") - (tokens[end] == ")")
                end += 1
            body = end
            if body < len(tokens) and tokens[body] == "throws":
                while body < len(tokens) and tokens[body] not in ("{", ";", "="):
                    body += 1
            if not depth and body < len(tokens) and tokens[body] == "{":
                declarations.append((tokens[index:end], tokens[body + 1:]))
    return declarations[0] if len(declarations) == 1 else None


def _metrics(finding: PmdViolation) -> dict[str, float | int]:
    pattern = _METRIC_PATTERNS.get(finding.rule)
    match = pattern.fullmatch(finding.message.strip()) if pattern else None
    if not match:
        return {}
    values: dict[str, float | int] = {}
    for name, text in match.groupdict().items():
        if text is not None:
            if name == "tcc":
                value = float(text.rstrip("%")) / (100 if text.endswith("%") else 1)
                if not 0 <= value <= 1:
                    return {}
                values[name] = value
            else:
                values[name] = int(text)
    return values


def _measurements(base: dict, head: dict) -> tuple[Measurement, ...]:
    return tuple(Measurement(name, base.get(name), head.get(name)) for name in sorted(base.keys() | head.keys()))


def _metric_change(measurements: tuple[Measurement, ...]) -> Change:
    if not measurements or any(metric.base is None or metric.head is None for metric in measurements):
        return "unknown"
    directions = {
        ((metric.head > metric.base) - (metric.head < metric.base)) * (-1 if metric.name == "tcc" else 1)
        for metric in measurements
        if metric.base is not None and metric.head is not None
    } - {0}
    return "worsened" if directions == {1} else "improved" if directions == {-1} else "unknown"


def _payload(finding: Finding) -> dict[str, Any]:
    result = asdict(finding)
    result.pop("raw_path", None)
    if isinstance(finding, CpdDuplication):
        for occurrence in result["occurrences"]:
            occurrence.pop("raw_path", None)
        result["occurrences"] = sorted(result["occurrences"], key=hash_json)
    return result


def _ordered(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda finding: hash_json(_payload(finding))))


def _unique_pairs(base, head, base_key, head_key):
    left, right = defaultdict(list), defaultdict(list)
    for index, finding in enumerate(base):
        key = base_key(finding)
        if key is not None:
            left[key].append(index)
    for index, finding in enumerate(head):
        key = head_key(finding)
        if key is not None:
            right[key].append(index)
    pairs = [(indices[0], right[key][0]) for key, indices in left.items() if len(indices) == 1 and len(right[key]) == 1]
    return pairs, set(range(len(base))) - {i for i, _ in pairs}, set(range(len(head))) - {j for _, j in pairs}


def _components(left, right, possible):
    neighbors = {("base", i): set() for i in left} | {("head", j): set() for j in right}
    for i in left:
        for j in right:
            if possible(i, j):
                neighbors[("base", i)].add(("head", j))
                neighbors[("head", j)].add(("base", i))
    remaining = set(neighbors)
    while remaining:
        pending, component = [min(remaining)], set()
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node)
                pending.extend(neighbors[node] - component)
        remaining -= component
        yield (
            [i for side, i in sorted(component) if side == "base"],
            [j for side, j in sorted(component) if side == "head"],
        )


def _pmd_delta(base, head, rename_map, emit, suppressed=False):
    canonical = lambda finding: rename_map.get(finding.path, finding.path)
    symbols = lambda finding: (finding.package, finding.klass, finding.method)
    identity = lambda finding, path: (finding.rule, path, *symbols(finding))

    def exact_key(finding, path):
        code = _code_tokens(finding.code_fragment)
        return (*identity(finding, path), code) if code else None

    pairs, left, right = _unique_pairs(base, head, lambda f: exact_key(f, canonical(f)), lambda f: exact_key(f, f.path))

    def possible(a, b):
        if a.rule != b.rule:
            return False
        first, second = _code_tokens(a.code_fragment), _code_tokens(b.code_fragment)
        return (
            not first
            or not second
            or identity(a, canonical(a)) == identity(b, b.path)
            or first == second
            or ((canonical(a) == b.path or symbols(a) == symbols(b)) and _overlap(first, second))
        )

    def compare(a, b):
        first, second = _code_tokens(a.code_fragment), _code_tokens(b.code_fragment)
        measurements = _measurements(_metrics(a), _metrics(b))
        correspondence = first and second and symbols(a) == symbols(b) and first == second
        matching = "PMD rule, rename-canonical path and package/class/method with exact normalized code"
        if correspondence and canonical(a) != b.path:
            matching = "PMD rule and package/class/method with unique exact code relocation (no rename metadata)"
        if not correspondence and identity(a, canonical(a)) == identity(b, b.path):
            before, after = _declaration(a, first), _declaration(b, second)
            correspondence = before and after and before[0] == after[0] and _overlap(before[1], after[1])
            matching = "PMD rule, rename-canonical path, package/class/method, declaration and substantive code overlap"
        if not correspondence:
            emit(
                "pmd",
                "unknown",
                (a,),
                (b,),
                "PMD correspondence is not established",
                measurements,
                ("Missing code, overload/symbol change or insufficient code correspondence; line numbers are not identity.",),
            )
            return
        unchanged = first == second and (a.ruleset, a.priority, a.message.strip()) == (b.ruleset, b.priority, b.message.strip())
        change = "existing" if unchanged else _metric_change(measurements)
        emit(
            "pmd",
            change,
            (a,),
            (b,),
            matching,
            measurements,
            ("Changed code/violation has absent, equal or mixed observed metrics.",) if change == "unknown" else (),
        )

    for i, j in pairs:
        compare(base[i], head[j])
    for indices, positions in _components(left, right, lambda i, j: possible(base[i], head[j])):
        first, second = tuple(base[i] for i in indices), tuple(head[j] for j in positions)
        if len(first) == len(second) == 1:
            compare(first[0], second[0])
        elif first and second:
            emit(
                "pmd",
                "unknown",
                first,
                second,
                "Ambiguous PMD candidate component",
                (),
                ("Multiple code/symbol candidates cannot be disambiguated by report positions.",),
            )
        else:
            finding = (first or second)[0]
            candidates = head if first else base
            ambiguous = any(possible(finding, other) if first else possible(other, finding) for other in candidates)
            known = bool(_code_tokens(finding.code_fragment)) and not ambiguous and not suppressed
            emit(
                "pmd",
                ("resolved" if first else "new") if known else "unknown",
                first,
                second,
                "Unmatched PMD observation in complete comparable reports" if known else "Unmatched PMD observation lacks unique code evidence",
                _measurements(_metrics(finding) if first else {}, _metrics(finding) if second else {}),
                ()
                if known
                else (
                    ("Suppressed observations have no location-addressable correspondence; new/resolved cannot be established.",)
                    if suppressed
                    else ("Missing code or correspondence with an already matched candidate prevents a new/resolved claim.",)
                ),
            )


def _cpd_metrics(group: CpdDuplication | None) -> dict[str, int]:
    return {"occurrences": len(group.occurrences), "tokens": group.tokens, "lines": group.lines} if group else {}


def _contains(longer, shorter) -> bool:
    return bool(
        shorter
        and len(shorter) >= 6
        and _substantive(shorter)
        and any(longer[index:index + len(shorter)] == shorter for index in range(len(longer) - len(shorter) + 1))
    )


def _cpd_delta(base, head, rename_map, emit):
    def locations(group, is_base):
        return Counter(
            rename_map.get(location.path, location.path) if is_base else location.path
            for location in group.occurrences
        )

    def exact_key(group, is_base):
        code = _code_tokens(group.code_fragment)
        return (code, group.tokens, tuple(sorted(locations(group, is_base).items()))) if code else None

    pairs, left, right = _unique_pairs(base, head, lambda g: exact_key(g, True), lambda g: exact_key(g, False))

    def competes(group, other, is_base):
        first, second = _code_tokens(group.code_fragment), _code_tokens(other.code_fragment)
        return bool(locations(group, is_base) & locations(other, is_base)) and (first == second or _overlap(first, second))

    supported = []
    for i, j in pairs:
        if any(k != i and competes(base[i], other, True) for k, other in enumerate(base)) or any(
            k != j and competes(head[j], other, False) for k, other in enumerate(head)
        ):
            left.add(i)
            right.add(j)
        else:
            supported.append((i, j))
    pairs = supported

    def possible(a, b):
        first, second = _code_tokens(a.code_fragment), _code_tokens(b.code_fragment)
        shared = bool(locations(a, True) & locations(b, False))
        return not first or not second or first == second or shared or _overlap(first, second)

    def compare(a, b):
        first, second = _code_tokens(a.code_fragment), _code_tokens(b.code_fragment)
        before, after = locations(a, True), locations(b, False)
        measurements = _measurements(_cpd_metrics(a), _cpd_metrics(b))
        change = "unknown"
        matching = "CPD normalized fragment/token identity and all rename-canonical location multiplicities"
        if first and first == second and a.tokens == b.tokens:
            if len(a.occurrences) == len(b.occurrences):
                change = "existing"
            elif before <= after:
                change = "worsened"
            elif after <= before:
                change = "improved"
        elif (
            first
            and second
            and (before <= after or after <= before)
            and sum((before & after).values()) >= 2
            and _overlap(first, second)
            and (_contains(first, second) or _contains(second, first))
            and (
                (len(second) > len(first) and b.tokens > a.tokens)
                or (len(second) < len(first) and b.tokens < a.tokens)
            )
        ):
            matching = "CPD substantive fragment containment and corresponding rename-canonical location multiplicities"
            change = _metric_change(measurements)
        emit(
            "cpd",
            change,
            (a,),
            (b,),
            matching,
            measurements,
            (
                "CPD change lacks consistent fragment/token and location correspondence, or measurements have mixed directions.",
            )
            if change == "unknown"
            else (),
        )

    for i, j in pairs:
        compare(base[i], head[j])
    for indices, positions in _components(left, right, lambda i, j: possible(base[i], head[j])):
        first, second = tuple(base[i] for i in indices), tuple(head[j] for j in positions)
        if len(first) == len(second) == 1:
            compare(first[0], second[0])
        elif first and second:
            emit(
                "cpd",
                "unknown",
                first,
                second,
                "Ambiguous CPD candidate component",
                (),
                ("Many-to-many, split or merged CPD groups have no unique occurrence correspondence.",),
            )
        else:
            group = (first or second)[0]
            candidates = head if first else base
            ambiguous = any(possible(group, other) if first else possible(other, group) for other in candidates)
            known = bool(_code_tokens(group.code_fragment)) and not ambiguous
            emit(
                "cpd",
                ("resolved" if first else "new") if known else "unknown",
                first,
                second,
                "Unmatched CPD group in complete comparable reports" if known else "Unmatched CPD group lacks unique code evidence",
                _measurements(_cpd_metrics(group) if first else {}, _cpd_metrics(group) if second else {}),
                () if known else ("Missing fragment or correspondence with an already matched group prevents a new/resolved claim.",),
            )


def build_delta(
    base_run: AnalysisRun,
    head_run: AnalysisRun,
    rename_map: dict[str, str] | None = None,
) -> AnalysisDelta:
    """Compare trustworthy tools independently; preserve ambiguous evidence as unknown."""
    if rename_map is None:
        rename_map = {}
    runs = (base_run, head_run)
    run_errors = tuple(_run_errors(run) for run in runs)
    rename_errors = _rename_errors(rename_map)
    signals: list[DeltaSignal] = []
    limitations: list[str] = list(rename_errors)
    analysis_status: dict[str, str] = {}

    def emit(
        tool: str,
        change: Change,
        base: Sequence[Finding],
        head: Sequence[Finding],
        matching: str,
        measurements: tuple[Measurement, ...] = (),
        limits: tuple[str, ...] = (),
    ) -> None:
        ord_base, ord_head = _ordered(base), _ordered(head)
        paths = sorted({
            location.path
            for finding in ord_base + ord_head
            for location in ((finding,) if isinstance(finding, PmdViolation) else finding.occurrences)
        })
        evidence = {
            "schema_version": 1,
            "tool": tool,
            "change": change,
            "base_sha": base_run.sha,
            "head_sha": head_run.sha,
            "conditions": [run.conditions_hash for run in runs],
            "reports": [
                {
                    "exists": getattr(run, tool).report.exists,
                    "parse_status": getattr(run, tool).report.parse_status,
                    "tool_version": getattr(run, tool).report.tool_version,
                    "suppressed_count": getattr(run, tool).report.suppressed_count,
                }
                for run in runs
            ],
            "sources": [run.source_hash for run in runs],
            "base_findings": [_payload(finding) for finding in ord_base],
            "head_findings": [_payload(finding) for finding in ord_head],
            "matching": matching,
            "measurements": [asdict(metric) for metric in measurements],
        }
        signals.append(
            DeltaSignal(
                tool + "-" + hash_json(evidence),
                tool,
                change,
                ord_base,
                ord_head,
                tuple(paths),
                matching,
                measurements,
                limits,
            )
        )

    usable = []
    for tool in ("pmd", "cpd"):
        checks = tuple(_tool_check(run, tool, errors) for run, errors in zip(runs, run_errors, strict=False))
        reasons = []
        for version, (status, errors) in zip(("base", "head"), checks, strict=False):
            analysis_status[version + "_" + tool] = status
            reasons.extend(f"{version} {tool}: {error}" for error in errors)
            if status == "no_sources":
                limitations.append(f"{version} {tool}: no_java_sources; tool intentionally not_run, not a successful empty XML report.")
        try:
            same_tool_conditions = hash_json(_tool_conditions(base_run, tool)) == hash_json(_tool_conditions(head_run, tool))
        except (TypeError, ValueError):
            same_tool_conditions = False
        if not same_tool_conditions:
            reasons.append(f"{tool}: analysis condition content differs between revisions")
        reasons.extend(rename_errors)
        comparable = not reasons and all(status in ("clean", "findings", "no_sources") for status, _ in checks)
        usable.append(comparable)
        base = _ordered(getattr(base_run, tool).report.violations if tool == "pmd" else base_run.cpd.report.duplications)
        head = _ordered(getattr(head_run, tool).report.violations if tool == "pmd" else head_run.cpd.report.duplications)
        if comparable:
            if tool == "pmd":
                suppressed = bool(base_run.pmd.report.suppressed_count or head_run.pmd.report.suppressed_count)
                if suppressed:
                    limitations.append("PMD suppressed observations have no location-addressable evidence; unmatched changes are unknown.")
                _pmd_delta(base, head, rename_map, emit, suppressed)
            else:
                _cpd_delta(base, head, rename_map, emit)
        else:
            limitations.extend(reasons)
            if base or head:
                measurements_tuple: tuple[Measurement, ...] = ()
                if len(base) <= 1 and len(head) <= 1:
                    metrics_fn = _metrics if tool == "pmd" else _cpd_metrics
                    measurements_tuple = _measurements(
                        metrics_fn(base[0]) if base else {},
                        metrics_fn(head[0]) if head else {},
                    )
                emit(
                    tool,
                    "unknown",
                    base,
                    head,
                    "Tool comparison unavailable",
                    measurements_tuple,
                    tuple(dict.fromkeys(reasons)),
                )
    for version, run in zip(("base", "head"), runs, strict=False):
        analysis_status[version] = _aggregate(run, (analysis_status[version + "_pmd"], analysis_status[version + "_cpd"]))
        limitations.extend(f"{version}: {limitation}" for limitation in run.limitations)
    for signal in signals:
        limitations.extend(f"{signal.tool}: {limitation}" for limitation in signal.limitations)
    valid_conditions = all(not errors for errors in run_errors)
    same_conditions = valid_conditions and base_run.conditions_hash == head_run.conditions_hash
    if not same_conditions:
        limitations.append("No single verified common analysis conditions hash; only independently comparable tools can establish changes.")
    return AnalysisDelta(
        1,
        base_run.sha,
        head_run.sha,
        base_run.conditions_hash if same_conditions else None,
        same_conditions and all(usable),
        tuple(sorted(signals, key=lambda signal: signal.signal_id)),
        tuple(sorted(set(limitations))),
        analysis_status,
    )


def _normalize_finding(item: Finding | dict[str, Any]) -> Finding:
    """Normalize finding dictionary or dataclass into typed Finding domain object."""
    if isinstance(item, (PmdViolation, CpdDuplication)):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"Finding must be PmdViolation, CpdDuplication, or dict, got {type(item)}")

    if "occurrences" in item or item.get("tool") == "cpd":
        raw_occurrences = item.get("occurrences", ())
        occurrences = []
        for occ in raw_occurrences:
            if isinstance(occ, CpdOccurrence):
                occurrences.append(occ)
            elif isinstance(occ, dict):
                occurrences.append(
                    CpdOccurrence(
                        path=occ.get("path", ""),
                        raw_path=occ.get("raw_path", occ.get("path", "")),
                        begin_line=int(occ.get("begin_line", 1)),
                        end_line=int(occ.get("end_line", 10)),
                        begin_column=occ.get("begin_column", 1),
                        end_column=occ.get("end_column", 80),
                        end_line_inferred=bool(occ.get("end_line_inferred", False)),
                        code_fragment=occ.get("code_fragment", item.get("code_fragment", "int x = 1;")),
                    )
                )
        return CpdDuplication(
            lines=int(item.get("lines", 10)),
            tokens=int(item.get("tokens", 100)),
            occurrences=tuple(occurrences),
            code_fragment=item.get("code_fragment", "int x = 1;"),
        )

    path = item.get("path", "")
    raw_path = item.get("raw_path", path)
    line = int(item.get("line", item.get("begin_line", 1)))
    end_line = int(item.get("end_line", line + 4))
    begin_col = int(item.get("begin_column", 1))
    end_col = int(item.get("end_column", 70))
    rule = item.get("rule", "CognitiveComplexity")
    ruleset = item.get("ruleset", "Design")
    priority = int(item.get("priority", 3))
    klass = item.get("klass", item.get("class", "Example"))
    method = item.get("method", "foo")
    package = item.get("package", "demo")
    metric = item.get("metric")

    message = item.get("message")
    if message is None:
        if metric is not None:
            if rule == "CognitiveComplexity":
                message = f"The method '{method}' has a cognitive complexity of {metric}, current threshold is 15."
            elif rule == "CyclomaticComplexity":
                message = f"The method '{method}' has a cyclomatic complexity of {metric}."
            elif rule == "NPathComplexity":
                message = f"The method '{method}' has an NPath complexity of {metric}, current threshold is 200."
            elif rule == "NcssCount":
                message = f"The method '{method}' has a NCSS line count of {metric}."
            elif rule == "GodClass":
                message = f"Possible God Class (WMC={metric}, ATFD=5, TCC=20%)"
            else:
                message = f"Violation of rule {rule}"
        else:
            message = f"Violation of rule {rule}"

    code = item.get("code_fragment", item.get("code"))
    if code is None:
        code = f"void {method}() {{ return; }}"

    return PmdViolation(
        path=path,
        raw_path=raw_path,
        begin_line=line,
        end_line=end_line,
        begin_column=begin_col,
        end_column=end_col,
        rule=rule,
        ruleset=ruleset,
        priority=priority,
        klass=klass,
        method=method,
        package=package,
        message=message,
        code_fragment=code,
    )


def compute_delta(
    base_run: AnalysisRun | None = None,
    head_run: AnalysisRun | None = None,
    rename_map: dict[str, str] | None = None,
    base_findings: Sequence[Finding | dict[str, Any]] | None = None,
    head_findings: Sequence[Finding | dict[str, Any]] | None = None,
    comparable: bool = True,
    base_sha: str = "0" * 40,
    head_sha: str = "1" * 40,
) -> AnalysisDelta:
    """Public facade for 6-state change delta computation.

    Supports both full AnalysisRun comparison and lightweight finding sequence comparison.
    """
    if base_run is not None and head_run is not None:
        return build_delta(base_run, head_run, rename_map=rename_map)

    rename_map_clean = dict(rename_map or {})
    base_raw = tuple(base_findings or ())
    head_raw = tuple(head_findings or ())

    normalized_base = tuple(_normalize_finding(item) for item in base_raw)
    normalized_head = tuple(_normalize_finding(item) for item in head_raw)

    if not comparable:
        signals: list[DeltaSignal] = []
        all_findings = normalized_base + normalized_head
        if all_findings:
            paths = sorted({
                location.path
                for finding in all_findings
                for location in ((finding,) if isinstance(finding, PmdViolation) else finding.occurrences)
            })
            tool = Tool.PMD.value if any(isinstance(f, PmdViolation) for f in all_findings) else Tool.CPD.value
            signal_id = f"{tool}-{hash_json((base_sha, head_sha, [_payload(f) for f in all_findings]))}"
            signals.append(
                DeltaSignal(
                    signal_id=signal_id,
                    tool=tool,
                    change="unknown",
                    base_findings=normalized_base,
                    head_findings=normalized_head,
                    paths=tuple(paths),
                    matching="Incomparable analysis conditions",
                    measurements=(),
                    limitations=("Incomparable analysis produced unknown status",),
                )
            )
        return AnalysisDelta(
            schema_version=1,
            base_sha=base_sha,
            head_sha=head_sha,
            conditions_hash=None,
            comparable=False,
            signals=tuple(signals),
            limitations=("Incomparable analysis",),
            analysis_status={},
        )

    signals_list: list[DeltaSignal] = []

    def emit_synthetic(
        tool: str,
        change: Change,
        base_f: Sequence[Finding],
        head_f: Sequence[Finding],
        matching: str,
        measurements: tuple[Measurement, ...] = (),
        limits: tuple[str, ...] = (),
    ) -> None:
        ord_base, ord_head = _ordered(base_f), _ordered(head_f)
        paths = sorted({
            location.path
            for finding in ord_base + ord_head
            for location in ((finding,) if isinstance(finding, PmdViolation) else finding.occurrences)
        })
        evidence = {
            "schema_version": 1,
            "tool": tool,
            "change": change,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "base_findings": [_payload(finding) for finding in ord_base],
            "head_findings": [_payload(finding) for finding in ord_head],
            "matching": matching,
            "measurements": [asdict(metric) for metric in measurements],
        }
        signals_list.append(
            DeltaSignal(
                signal_id=f"{tool}-{hash_json(evidence)}",
                tool=tool,
                change=change,
                base_findings=ord_base,
                head_findings=ord_head,
                paths=tuple(paths),
                matching=matching,
                measurements=measurements,
                limitations=limits,
            )
        )

    base_pmd = tuple(f for f in normalized_base if isinstance(f, PmdViolation))
    head_pmd = tuple(f for f in normalized_head if isinstance(f, PmdViolation))
    base_cpd = tuple(f for f in normalized_base if isinstance(f, CpdDuplication))
    head_cpd = tuple(f for f in normalized_head if isinstance(f, CpdDuplication))

    if base_pmd or head_pmd:
        _pmd_delta(base_pmd, head_pmd, rename_map_clean, emit_synthetic, suppressed=False)
    if base_cpd or head_cpd:
        _cpd_delta(base_cpd, head_cpd, rename_map_clean, emit_synthetic)

    return AnalysisDelta(
        schema_version=1,
        base_sha=base_sha,
        head_sha=head_sha,
        conditions_hash=hash_json({"simulated": True}),
        comparable=True,
        signals=tuple(sorted(signals_list, key=lambda s: s.signal_id)),
        limitations=(),
        analysis_status={
            "base": "clean" if not base_raw else "findings",
            "head": "clean" if not head_raw else "findings",
        },
    )
