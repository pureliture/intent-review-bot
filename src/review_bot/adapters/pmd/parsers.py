"""Strict XML report parsers for PMD 7.3.0 and CPD with complete XXE immunity."""
from __future__ import annotations

import hashlib
import stat
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from review_bot.domain.models_v1 import (
    AnalysisError,
    CpdDuplication,
    CpdOccurrence,
    ParsedReport,
    PmdViolation,
)

NS = {"pmd": "http://pmd.sourceforge.net/report/2.0.0"}
CPD_NS = {"cpd": "https://pmd-code.org/schema/cpd-report"}
MAX_STRICT_REPORT_BYTES = 64 * 1024 * 1024  # 64 MiB limit


@dataclass(frozen=True)
class PmdReport(ParsedReport):
    """Strongly typed parsed PMD static analysis report."""


@dataclass(frozen=True)
class CpdReport(ParsedReport):
    """Strongly typed parsed CPD token duplication report."""


def short_path(full: str) -> str:
    """Strip standard Java source root prefixes."""
    if "main/java/" in full:
        return full.split("main/java/", 1)[1]
    return full


class _StrictTreeBuilder(ET.TreeBuilder):
    """ElementTree TreeBuilder strictly prohibiting DTD and entity declarations (XXE immunity)."""

    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        raise ValueError("DTD and entity declarations are not allowed")


def parse_xml(path: Path) -> list[dict]:
    """Legacy dictionary parser for PMD XML reports with XXE immunity."""
    if path.is_symlink():
        raise ValueError("Report must not be a symbolic link")
    parser = ET.XMLParser(target=_StrictTreeBuilder())
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()
    files: list[dict] = []
    for file_elem in root.findall("pmd:file", NS) or root.findall("file"):
        raw = file_elem.get("name", "")
        violations = []
        for v in file_elem.findall("pmd:violation", NS) or file_elem.findall("violation"):
            violations.append(
                {
                    "line": int(v.get("beginline", "0")),
                    "rule": v.get("rule", ""),
                    "priority": int(v.get("priority", "5")),
                    "method": v.get("method", ""),
                    "klass": v.get("class", ""),
                    "package": v.get("package", ""),
                    "msg": (v.text or "").strip(),
                }
            )
        if violations:
            files.append(
                {
                    "path": short_path(raw),
                    "raw": raw,
                    "violations": violations,
                }
            )
    return files


def parse_cpd(path: Path) -> dict:
    """Legacy dictionary parser for CPD XML reports with XXE immunity."""
    if not path.exists():
        return {"blocks": [], "total_blocks": 0, "total_lines": 0}
    if path.is_symlink():
        raise ValueError("Report must not be a symbolic link")

    parser = ET.XMLParser(target=_StrictTreeBuilder())
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()
    blocks = []
    total_lines = 0
    dups = root.findall("cpd:duplication", CPD_NS) or root.findall("duplication")
    for d in dups:
        lines_c = int(d.get("lines", "0"))
        tokens = int(d.get("tokens", "0"))
        total_lines += lines_c
        occurrences = []
        files = d.findall("cpd:file", CPD_NS) or d.findall("file")
        for f in files:
            occurrences.append(
                {
                    "path": short_path(f.get("path", "")),
                    "line": int(f.get("line", "0")),
                }
            )
        blocks.append({"lines": lines_c, "tokens": tokens, "occurrences": occurrences})
    return {"blocks": blocks, "total_blocks": len(blocks), "total_lines": total_lines}


def filter_cpd_for_changed(cpd: dict, changed_norm: set[str]) -> list[dict]:
    """Extract CPD duplication blocks touching changed files."""
    out: list[dict] = []
    for blk in cpd.get("blocks", []):
        changed_occs = [o for o in blk["occurrences"] if o["path"] in changed_norm]
        if changed_occs:
            others = [o for o in blk["occurrences"] if o["path"] not in changed_norm]
            out.append(
                {
                    "lines": blk["lines"],
                    "tokens": blk["tokens"],
                    "changed": changed_occs,
                    "others": others,
                }
            )
    out.sort(key=lambda b: -b["lines"])
    return out


def _read_and_validate_payload(content_or_path: str | bytes | Path) -> tuple[bytes, str]:
    """Normalize input into raw bytes and path string with strict size and symlink checks."""
    if isinstance(content_or_path, Path) or (
        isinstance(content_or_path, str)
        and not content_or_path.strip().startswith("<")
        and len(content_or_path) < 4096
        and Path(content_or_path).exists()
    ):
        path = Path(content_or_path)
        if path.is_symlink():
            raise ValueError("Report must not be a symbolic link")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Report must be a regular file")
        if metadata.st_size > MAX_STRICT_REPORT_BYTES:
            raise ValueError("Report exceeds the strict parser size limit")
        with path.open("rb") as stream:
            data = stream.read(MAX_STRICT_REPORT_BYTES + 1)
        if len(data) > MAX_STRICT_REPORT_BYTES:
            raise ValueError("Report exceeds the strict parser size limit")
        return data, str(path)

    if isinstance(content_or_path, str):
        data = content_or_path.encode("utf-8")
    elif isinstance(content_or_path, bytes):
        data = content_or_path
    else:
        raise TypeError(f"Unsupported report payload type: {type(content_or_path)}")

    if len(data) > MAX_STRICT_REPORT_BYTES:
        raise ValueError("Report exceeds the strict parser size limit")
    return data, "<memory>"


def _source_path(raw: str, source_root: Path | None, *, lexical: bool = False) -> str:
    """Sanitize report path against directory traversal, Windows separators, and escapes."""
    path = Path(raw)
    if (
        not raw
        or not path.name
        or ".." in path.parts
        or "\\" in raw
        or ":" in raw
        or raw.startswith(("~", "//"))
        or any(ord(c) < 32 or ord(c) == 127 for c in raw)
    ):
        raise ValueError("Report references an unsafe source path")

    if source_root is None:
        return path.as_posix()

    root = source_root.absolute() if lexical else source_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative = (candidate.absolute() if lexical else candidate.resolve()).relative_to(root)
    except ValueError:
        raise ValueError("Report references code outside its revision snapshot") from None

    if not relative.name:
        raise ValueError("Report references the source root rather than a source file")

    return relative.as_posix()


def _source_fragment(
    path: str,
    begin: int,
    end: int,
    source_root: Path | None,
    source_contents: dict[str, str] | None = None,
) -> str | None:
    """Extract multi-line source fragment in-memory or from filesystem."""
    if source_contents is not None:
        if path not in source_contents or not isinstance(source_contents[path], str):
            raise ValueError("Report references code absent from the fixed source data")
        lines = source_contents[path].splitlines(keepends=True)
    elif source_root is not None:
        target = source_root / path
        if not target.is_file():
            return None
        lines = target.read_bytes().decode("utf-8").splitlines(keepends=True)
    else:
        return None

    if end > len(lines):
        raise ValueError("Report span exceeds the source file")
    return "".join(lines[begin - 1 : end])


def _attributes(element: ET.Element, allowed: set[str]) -> None:
    """Ensure element attributes are within allowed set."""
    if element.attrib.keys() - allowed:
        raise ValueError("Unexpected report attribute or namespace")


def _children(
    element: ET.Element,
    ns_prefix: str,
    allowed: Sequence[str],
    *,
    text: bool = False,
) -> None:
    """Validate child element tags, strict order, and absence of unexpected text."""
    if not text and (element.text or "").strip():
        raise ValueError("Unexpected report text")

    positions = {
        f"{{{ns_prefix}}}{name}" if ns_prefix else name: index
        for index, name in enumerate(allowed)
    }
    previous = -1
    for child in element:
        tag_key = child.tag
        if tag_key not in positions or positions[tag_key] < previous:
            raise ValueError("Unexpected report element, order or namespace")
        previous = positions[tag_key]
        if (child.tail or "").strip():
            raise ValueError("Unexpected report text")


def _nonnegative_int(element: ET.Element, attribute: str) -> int:
    raw = element.get(attribute)
    if raw is None or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"Invalid or missing nonnegative integer: {attribute}")
    return int(raw)


def _positive_int(element: ET.Element, attribute: str) -> int:
    val = _nonnegative_int(element, attribute)
    if val < 1:
        raise ValueError(f"Invalid or missing positive integer: {attribute}")
    return val


def _required(element: ET.Element, attribute: str) -> str:
    val = element.get(attribute)
    if not val or not val.strip():
        raise ValueError(f"Missing report attribute: {attribute}")
    return val


def _reported_error(
    element: ET.Element,
    ns_prefix: str,
    kind: str,
    source_root: Path | None,
    *,
    lexical: bool = False,
) -> AnalysisError:
    if kind == "configuration_error":
        _attributes(element, {"rule", "msg"})
        _children(element, ns_prefix, ())
        return AnalysisError(kind, _required(element, "msg"), detail=f"rule={_required(element, 'rule')}")
    _attributes(element, {"filename", "msg"})
    _children(element, ns_prefix, (), text=True)
    return AnalysisError(
        kind,
        _required(element, "msg"),
        _source_path(_required(element, "filename"), source_root, lexical=lexical),
        element.text or "",
    )


def _detect_root_and_namespace(
    root: ET.Element,
    tool: str,
    require_namespace: bool,
) -> tuple[str, str]:
    """Validate root element and determine namespace prefix.

    Returns:
        (namespace_uri, version_attribute_name)
    """
    expected_ns = NS["pmd"] if tool == "pmd" else CPD_NS["cpd"]
    expected_tag_name = "pmd" if tool == "pmd" else "pmd-cpd"
    namespaced_tag = f"{{{expected_ns}}}{expected_tag_name}"

    if root.tag == namespaced_tag:
        ns_uri = expected_ns
    elif root.tag == expected_tag_name:
        if require_namespace:
            raise ValueError(f"Expected {tool.upper()} report namespace '{expected_ns}'")
        ns_uri = ""
    else:
        raise ValueError(f"Expected {tool.upper()} report root and namespace")

    version_attr = "version" if tool == "pmd" else "pmdVersion"
    return ns_uri, version_attr


def parse_pmd_xml(
    content_or_path: str | bytes | Path,
    *,
    source_root: Path | None = None,
    source_contents: dict[str, str] | None = None,
    raise_on_error: bool = True,
    require_namespace: bool = False,
) -> PmdReport:
    """Parse PMD XML report into strongly typed PmdReport with complete XXE immunity."""
    report_path = str(content_or_path) if isinstance(content_or_path, (str, Path)) else "<memory>"
    try:
        data, path_str = _read_and_validate_payload(content_or_path)
    except Exception as exc:
        if raise_on_error:
            raise
        return PmdReport(
            tool="pmd",
            path=report_path,
            exists=False,
            parse_status="invalid",
            errors=(AnalysisError("invalid_report", str(exc), report_path),),
        )

    content_hash = hashlib.sha256(data).hexdigest()
    try:
        parser = ET.XMLParser(target=_StrictTreeBuilder())
        root = ET.fromstring(data, parser=parser)
    except Exception as exc:
        if raise_on_error:
            raise
        return PmdReport(
            tool="pmd",
            path=path_str,
            exists=True,
            content_hash=content_hash,
            parse_status="invalid",
            errors=(AnalysisError("invalid_report", str(exc), path_str),),
        )

    try:
        ns_uri, version_attr = _detect_root_and_namespace(root, "pmd", require_namespace)
        allowed_attrs = {"version", "timestamp"}
        if ns_uri:
            allowed_attrs.add("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
        _attributes(root, allowed_attrs)
        tool_version = _required(root, version_attr)

        violations: list[PmdViolation] = []
        errors: list[AnalysisError] = []
        suppressed_count = 0
        lexical = source_contents is not None

        _children(root, ns_uri, ("file", "error", "suppressedviolation", "configerror"))

        file_tag = f"{{{ns_uri}}}file" if ns_uri else "file"
        error_tag = f"{{{ns_uri}}}error" if ns_uri else "error"
        suppressed_tag = f"{{{ns_uri}}}suppressedviolation" if ns_uri else "suppressedviolation"
        config_error_tag = f"{{{ns_uri}}}configerror" if ns_uri else "configerror"
        violation_tag = f"{{{ns_uri}}}violation" if ns_uri else "violation"

        for element in root:
            if element.tag == error_tag:
                errors.append(_reported_error(element, ns_uri, "processing_error", source_root, lexical=lexical))
            elif element.tag == config_error_tag:
                errors.append(_reported_error(element, ns_uri, "configuration_error", source_root))
            elif element.tag == suppressed_tag:
                _attributes(element, {"filename", "suppressiontype", "msg", "usermsg"})
                _children(element, ns_uri, ())
                _source_path(_required(element, "filename"), source_root, lexical=lexical)
                _required(element, "suppressiontype")
                _required(element, "msg")
                if "usermsg" not in element.attrib:
                    raise ValueError("Missing report attribute: usermsg")
                suppressed_count += 1
            elif element.tag == file_tag:
                _attributes(element, {"name"})
                _children(element, ns_uri, ("violation",))
                if not len(element):
                    raise ValueError("PMD file needs at least one violation")
                raw = _required(element, "name")
                source_path = _source_path(raw, source_root, lexical=lexical)
                for violation in element:
                    if violation.tag != violation_tag:
                        raise ValueError("Unexpected report element, order or namespace")
                    _attributes(
                        violation,
                        {
                            "beginline", "endline", "begincolumn", "endcolumn",
                            "rule", "ruleset", "priority", "class", "method",
                            "package", "variable", "externalInfoUrl",
                        },
                    )
                    _children(violation, ns_uri, (), text=True)
                    begin, end = _positive_int(violation, "beginline"), _positive_int(violation, "endline")
                    begin_column = _positive_int(violation, "begincolumn")
                    end_column = _positive_int(violation, "endcolumn")
                    priority = _positive_int(violation, "priority")
                    if end < begin or (end == begin and end_column < begin_column) or priority > 5:
                        raise ValueError("Invalid PMD source span or priority")
                    message = (violation.text or "").strip()
                    if not message:
                        raise ValueError("Missing PMD violation message")

                    violations.append(
                        PmdViolation(
                            path=source_path,
                            raw_path=raw,
                            begin_line=begin,
                            end_line=end,
                            begin_column=begin_column,
                            end_column=end_column,
                            rule=_required(violation, "rule"),
                            ruleset=_required(violation, "ruleset"),
                            priority=priority,
                            klass=violation.get("class", ""),
                            method=violation.get("method", ""),
                            package=violation.get("package", ""),
                            message=message,
                            code_fragment=_source_fragment(source_path, begin, end, source_root, source_contents),
                        )
                    )

        return PmdReport(
            tool="pmd",
            path=path_str,
            exists=True,
            parse_status="valid",
            tool_version=tool_version,
            content_hash=content_hash,
            violations=tuple(violations),
            duplications=(),
            errors=tuple(errors),
            suppressed_count=suppressed_count,
        )
    except Exception as exc:
        if raise_on_error:
            raise
        return PmdReport(
            tool="pmd",
            path=path_str,
            exists=True,
            content_hash=content_hash,
            parse_status="invalid",
            errors=(AnalysisError("invalid_report", str(exc), path_str),),
        )


def parse_cpd_xml(
    content_or_path: str | bytes | Path,
    *,
    source_root: Path | None = None,
    source_contents: dict[str, str] | None = None,
    raise_on_error: bool = True,
    require_namespace: bool = False,
) -> CpdReport:
    """Parse CPD XML report into strongly typed CpdReport with complete XXE immunity."""
    report_path = str(content_or_path) if isinstance(content_or_path, (str, Path)) else "<memory>"
    try:
        data, path_str = _read_and_validate_payload(content_or_path)
    except Exception as exc:
        if raise_on_error:
            raise
        return CpdReport(
            tool="cpd",
            path=report_path,
            exists=False,
            parse_status="invalid",
            errors=(AnalysisError("invalid_report", str(exc), report_path),),
        )

    content_hash = hashlib.sha256(data).hexdigest()
    try:
        parser = ET.XMLParser(target=_StrictTreeBuilder())
        root = ET.fromstring(data, parser=parser)
    except Exception as exc:
        if raise_on_error:
            raise
        return CpdReport(
            tool="cpd",
            path=path_str,
            exists=True,
            content_hash=content_hash,
            parse_status="invalid",
            errors=(AnalysisError("invalid_report", str(exc), path_str),),
        )

    try:
        ns_uri, _ = _detect_root_and_namespace(root, "cpd", require_namespace)
        allowed_attrs = {"version", "pmdVersion", "timestamp"}
        if ns_uri:
            allowed_attrs.add("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
        _attributes(root, allowed_attrs)
        tool_version = _required(root, "pmdVersion")
        if "version" in root.attrib and root.get("version") != "1.0.0":
            raise ValueError("Unsupported CPD report schema version")

        duplications: list[CpdDuplication] = []
        errors: list[AnalysisError] = []
        lexical = source_contents is not None

        _children(root, ns_uri, ("file", "duplication", "error"))

        file_tag = f"{{{ns_uri}}}file" if ns_uri else "file"
        dup_tag = f"{{{ns_uri}}}duplication" if ns_uri else "duplication"
        err_tag = f"{{{ns_uri}}}error" if ns_uri else "error"
        code_tag = f"{{{ns_uri}}}codefragment" if ns_uri else "codefragment"

        for element in root:
            if element.tag == err_tag:
                errors.append(_reported_error(element, ns_uri, "processing_error", source_root, lexical=lexical))
            elif element.tag == file_tag:
                _attributes(element, {"path", "totalNumberOfTokens"})
                _children(element, ns_uri, ())
                _source_path(_required(element, "path"), source_root, lexical=lexical)
                _nonnegative_int(element, "totalNumberOfTokens")
            elif element.tag == dup_tag:
                _attributes(element, {"lines", "tokens"})
                _children(element, ns_uri, ("file", "codefragment"))
                lines, tokens = _positive_int(element, "lines"), _positive_int(element, "tokens")
                occurrences: list[CpdOccurrence] = []

                for occurrence in element.findall(file_tag):
                    _attributes(
                        occurrence,
                        {
                            "path", "line", "endline", "column", "begincolumn",
                            "endcolumn", "begintoken", "endtoken",
                        },
                    )
                    _children(occurrence, ns_uri, ())
                    raw = _required(occurrence, "path")
                    source_path = _source_path(raw, source_root, lexical=lexical)
                    begin = _positive_int(occurrence, "line")
                    end = (
                        _positive_int(occurrence, "endline")
                        if "endline" in occurrence.attrib
                        else begin + lines - 1
                    )
                    begin_attr = "column" if "column" in occurrence.attrib else "begincolumn"
                    begin_column = (
                        _positive_int(occurrence, begin_attr) if begin_attr in occurrence.attrib else None
                    )
                    end_column = (
                        _positive_int(occurrence, "endcolumn") if "endcolumn" in occurrence.attrib else None
                    )
                    if (
                        "column" in occurrence.attrib
                        and "begincolumn" in occurrence.attrib
                        and begin_column != _positive_int(occurrence, "begincolumn")
                    ):
                        raise ValueError("Conflicting CPD begin columns")
                    begin_token = (
                        _nonnegative_int(occurrence, "begintoken")
                        if "begintoken" in occurrence.attrib
                        else None
                    )
                    end_token = (
                        _nonnegative_int(occurrence, "endtoken")
                        if "endtoken" in occurrence.attrib
                        else None
                    )
                    if begin_token is not None and end_token is not None and end_token < begin_token:
                        raise ValueError("Invalid CPD token span")
                    if end < begin or (
                        end == begin
                        and begin_column is not None
                        and end_column is not None
                        and end_column < begin_column
                    ):
                        raise ValueError("Invalid CPD source span")

                    occurrences.append(
                        CpdOccurrence(
                            path=source_path,
                            raw_path=raw,
                            begin_line=begin,
                            end_line=end,
                            begin_column=begin_column,
                            end_column=end_column,
                            end_line_inferred="endline" not in occurrence.attrib,
                            code_fragment=_source_fragment(source_path, begin, end, source_root, source_contents),
                        )
                    )

                fragments = element.findall(code_tag)
                if len(occurrences) < 2 or len(fragments) != 1 or not (fragments[0].text or "").strip():
                    raise ValueError("CPD group needs at least two locations and one code fragment")
                if len({(o.path, o.begin_line, o.begin_column) for o in occurrences}) != len(occurrences):
                    raise ValueError("CPD group repeats the same source location")

                _attributes(fragments[0], set())
                _children(fragments[0], ns_uri, (), text=True)
                duplications.append(
                    CpdDuplication(
                        lines=lines,
                        tokens=tokens,
                        occurrences=tuple(occurrences),
                        code_fragment=fragments[0].text or "",
                    )
                )

        return CpdReport(
            tool="cpd",
            path=path_str,
            exists=True,
            parse_status="valid",
            tool_version=tool_version,
            content_hash=content_hash,
            violations=(),
            duplications=tuple(duplications),
            errors=tuple(errors),
        )
    except Exception as exc:
        if raise_on_error:
            raise
        return CpdReport(
            tool="cpd",
            path=path_str,
            exists=True,
            content_hash=content_hash,
            parse_status="invalid",
            errors=(AnalysisError("invalid_report", str(exc), path_str),),
        )


def parse_xml_strict(
    path: Path | str,
    *,
    source_root: Path | None = None,
    source_contents: dict[str, str] | None = None,
) -> ParsedReport:
    """Strict PMD parser returning ParsedReport with encapsulated errors."""
    return parse_pmd_xml(
        path,
        source_root=source_root,
        source_contents=source_contents,
        raise_on_error=False,
        require_namespace=True,
    )


def parse_cpd_strict(
    path: Path | str,
    *,
    source_root: Path | None = None,
    source_contents: dict[str, str] | None = None,
) -> ParsedReport:
    """Strict CPD parser returning ParsedReport with encapsulated errors."""
    return parse_cpd_xml(
        path,
        source_root=source_root,
        source_contents=source_contents,
        raise_on_error=False,
        require_namespace=True,
    )
