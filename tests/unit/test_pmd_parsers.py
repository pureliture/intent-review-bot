"""Unit tests for PMD and CPD strict XML report parsers."""
from __future__ import annotations

from pathlib import Path

import pytest

from review_bot.adapters.pmd.parsers import (
    MAX_STRICT_REPORT_BYTES,
    CpdReport,
    PmdReport,
    filter_cpd_for_changed,
    parse_cpd,
    parse_cpd_strict,
    parse_cpd_xml,
    parse_pmd_xml,
    parse_xml,
    parse_xml_strict,
    short_path,
)
from review_bot.domain.models_v1 import ParsedReport

VALID_PMD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0" version="7.3.0" timestamp="2026-10-06T12:00:00Z">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="10" endline="25" begincolumn="5" endcolumn="15"
               rule="CyclomaticComplexity" ruleset="Design" priority="3"
               class="Foo" method="bar" package="demo">
      Method bar has cyclomatic complexity of 12 (threshold is 10)
    </violation>
  </file>
</pmd>
"""

VALID_CPD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<pmd-cpd xmlns="https://pmd-code.org/schema/cpd-report" version="1.0.0" pmdVersion="7.3.0" timestamp="2026-10-06T12:00:00Z">
  <duplication lines="15" tokens="110">
    <file path="src/main/java/demo/A.java" line="10" endline="24" begintoken="1" endtoken="110"/>
    <file path="src/main/java/demo/B.java" line="30" endline="44" begintoken="1" endtoken="110"/>
    <codefragment><![CDATA[// duplicate code snippet]]></codefragment>
  </duplication>
</pmd-cpd>
"""

BARE_CPD_XML = """<?xml version="1.0" encoding="UTF-8"?>
<pmd-cpd version="1.0.0" pmdVersion="7.3.0" timestamp="2026-10-06T12:00:00Z">
  <duplication lines="10" tokens="105">
    <file path="src/main/java/demo/A.java" line="5" begintoken="10" endtoken="114"/>
    <file path="src/main/java/demo/B.java" line="15" begintoken="10" endtoken="114"/>
    <codefragment><![CDATA[some duplicate]]></codefragment>
  </duplication>
</pmd-cpd>
"""

MALICIOUS_XXE_DTD = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE test [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation rule="Test" ruleset="Test" priority="1">&xxe;</violation>
  </file>
</pmd>
"""

BILLION_LAUGHS_XML = """<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ELEMENT lolz (#PCDATA)>
 <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation rule="Test" ruleset="Test" priority="1">&lol1;</violation>
  </file>
</pmd>
"""


class TestPmdXmlParsing:
    def test_parse_valid_pmd(self) -> None:
        report = parse_pmd_xml(VALID_PMD_XML)
        assert isinstance(report, PmdReport)
        assert isinstance(report, ParsedReport)
        assert report.tool == "pmd"
        assert report.tool_version == "7.3.0"
        assert report.parse_status == "valid"
        assert report.finding_count == 1
        assert len(report.violations) == 1

        v = report.violations[0]
        assert v.rule == "CyclomaticComplexity"
        assert v.priority == 3
        assert v.begin_line == 10
        assert v.end_line == 25
        assert v.begin_column == 5
        assert v.end_column == 15
        assert v.klass == "Foo"
        assert v.method == "bar"
        assert v.package == "demo"
        assert "cyclomatic complexity" in v.message

    def test_parse_suppressed_violations(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0" version="7.3.0">
  <suppressedviolation filename="src/main/java/demo/Foo.java"
                       suppressiontype="annotation"
                       msg="Suppressed"
                       usermsg="Expected exception"/>
</pmd>
"""
        report = parse_pmd_xml(xml)
        assert report.suppressed_count == 1
        assert len(report.violations) == 0

    def test_parse_error_elements(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0" version="7.3.0">
  <error filename="src/main/java/demo/Broken.java" msg="ParseException: syntax error">
    Detailed stacktrace here
  </error>
  <configerror rule="FooRule" msg="Rule not configured properly"/>
</pmd>
"""
        report = parse_pmd_xml(xml)
        assert len(report.errors) == 2
        assert report.errors[0].kind == "processing_error"
        assert report.errors[0].message == "ParseException: syntax error"
        assert report.errors[1].kind == "configuration_error"
        assert report.errors[1].message == "Rule not configured properly"

    def test_xxe_rejection(self) -> None:
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(MALICIOUS_XXE_DTD)

    def test_billion_laughs_rejection(self) -> None:
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(BILLION_LAUGHS_XML)

    def test_size_limit_rejection(self) -> None:
        oversized = b"<pmd>" + b" " * (MAX_STRICT_REPORT_BYTES + 10) + b"</pmd>"
        with pytest.raises(ValueError, match="Report exceeds the strict parser size limit"):
            parse_pmd_xml(oversized)

    def test_invalid_line_span_rejected(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0" version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="20" endline="10" begincolumn="1" endcolumn="1"
               rule="Test" ruleset="Test" priority="1">
      Inverted range
    </violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Invalid PMD source span or priority"):
            parse_pmd_xml(xml)

    def test_invalid_priority_rejected(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0" version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="10" endline="20" begincolumn="1" endcolumn="1"
               rule="Test" ruleset="Test" priority="6">
      Priority 6 out of range
    </violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Invalid PMD source span or priority"):
            parse_pmd_xml(xml)


class TestCpdXmlParsing:
    def test_parse_valid_cpd_namespaced(self) -> None:
        report = parse_cpd_xml(VALID_CPD_XML)
        assert isinstance(report, CpdReport)
        assert isinstance(report, ParsedReport)
        assert report.tool == "cpd"
        assert report.tool_version == "7.3.0"
        assert report.finding_count == 1
        assert len(report.duplications) == 1

        d = report.duplications[0]
        assert d.lines == 15
        assert d.tokens == 110
        assert len(d.occurrences) == 2
        assert d.occurrences[0].path == "src/main/java/demo/A.java"
        assert d.occurrences[0].begin_line == 10
        assert d.occurrences[0].end_line == 24
        assert not d.occurrences[0].end_line_inferred

    def test_parse_bare_cpd_with_inferred_endline(self) -> None:
        report = parse_cpd_xml(BARE_CPD_XML)
        assert len(report.duplications) == 1
        d = report.duplications[0]
        assert d.tokens == 105
        assert len(d.occurrences) == 2

        # endline was absent, lines=10, begin_line=5 -> end_line = 5 + 10 - 1 = 14
        occ = d.occurrences[0]
        assert occ.begin_line == 5
        assert occ.end_line == 14
        assert occ.end_line_inferred

    def test_cpd_insufficient_occurrences_rejected(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd-cpd version="1.0.0" pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/demo/A.java" line="5" endline="15"/>
    <codefragment>snippet</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group needs at least two locations"):
            parse_cpd_xml(xml)

    def test_cpd_duplicate_location_rejected(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd-cpd version="1.0.0" pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/demo/A.java" line="5" endline="15"/>
    <file path="src/main/java/demo/A.java" line="5" endline="15"/>
    <codefragment>snippet</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group repeats the same source location"):
            parse_cpd_xml(xml)


class TestPathSanitizationAndBoundaries:
    @pytest.mark.parametrize(
        "bad_path",
        [
            "../outside.java",
            "foo/../../bar.java",
            "C:\\foo\\bar.java",
            "foo\\bar.java",
            "~/foo.java",
            "//network/foo.java",
            "foo:bar.java",
            "Foo\n.java",
            "Foo\t.java",
        ],
    )
    def test_unsafe_paths_rejected(self, bad_path: str) -> None:
        from xml.sax.saxutils import quoteattr
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<pmd version="7.3.0">
  <file name={quoteattr(bad_path)}>
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="Test" ruleset="Test" priority="1">
      msg
    </violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            parse_pmd_xml(xml)

    def test_out_of_snapshot_escape_rejected(self, tmp_path: Path) -> None:
        source_root = tmp_path / "root"
        source_root.mkdir()
        outside = tmp_path / "outside.java"
        outside.write_text("class Outside {}", encoding="utf-8")

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<pmd version="7.3.0">
  <file name="{outside}">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="Test" ruleset="Test" priority="1">
      msg
    </violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references code outside its revision snapshot"):
            parse_pmd_xml(xml, source_root=source_root)

    def test_symlink_report_file_rejected(self, tmp_path: Path) -> None:
        real_file = tmp_path / "real.xml"
        real_file.write_text(VALID_PMD_XML, encoding="utf-8")
        symlink_file = tmp_path / "symlink.xml"
        symlink_file.symlink_to(real_file)

        with pytest.raises(ValueError, match="Report must not be a symbolic link"):
            parse_pmd_xml(symlink_file)


class TestInMemSourceFragments:
    def test_fragment_extraction_from_dict(self) -> None:
        contents = {
            "src/main/java/demo/Foo.java": "line1\nline2\nline3\nline4\nline5\n",
        }
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="2" endline="4" begincolumn="1" endcolumn="1"
               rule="Test" ruleset="Test" priority="1">
      msg
    </violation>
  </file>
</pmd>
"""
        report = parse_pmd_xml(xml, source_contents=contents)
        assert report.violations[0].code_fragment == "line2\nline3\nline4\n"

    def test_fragment_span_exceeds_file_rejected(self) -> None:
        contents = {
            "src/main/java/demo/Foo.java": "line1\n",
        }
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="5" begincolumn="1" endcolumn="1"
               rule="Test" ruleset="Test" priority="1">
      msg
    </violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report span exceeds the source file"):
            parse_pmd_xml(xml, source_contents=contents)


class TestCompatibilityWrappers:
    def test_parse_xml_strict_encapsulates_errors(self, tmp_path: Path) -> None:
        malicious = tmp_path / "xxe.xml"
        malicious.write_text(MALICIOUS_XXE_DTD, encoding="utf-8")
        report = parse_xml_strict(malicious)
        assert report.parse_status == "invalid"
        assert len(report.errors) == 1
        assert "DTD and entity declarations" in report.errors[0].message

    def test_parse_cpd_strict_encapsulates_errors(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.xml"
        report = parse_cpd_strict(missing)
        assert report.parse_status == "missing" or report.parse_status == "invalid"
        assert len(report.errors) >= 1

    def test_legacy_helpers(self, tmp_path: Path) -> None:
        assert short_path("foo/src/main/java/demo/Bar.java") == "demo/Bar.java"
        assert short_path("foo/bar.txt") == "foo/bar.txt"

        pmd_file = tmp_path / "pmd.xml"
        pmd_file.write_text(VALID_PMD_XML, encoding="utf-8")
        parsed = parse_xml(pmd_file)
        assert len(parsed) == 1
        assert parsed[0]["path"] == "demo/Foo.java"

        cpd_file = tmp_path / "cpd.xml"
        cpd_file.write_text(VALID_CPD_XML, encoding="utf-8")
        parsed_cpd = parse_cpd(cpd_file)
        assert parsed_cpd["total_blocks"] == 1
        assert parsed_cpd["total_lines"] == 15

        changed = filter_cpd_for_changed(parsed_cpd, {"demo/A.java"})
        assert len(changed) == 1

    def test_legacy_helpers_reject_xxe_and_symlinks(self, tmp_path: Path) -> None:
        malicious = tmp_path / "xxe.xml"
        malicious.write_text(MALICIOUS_XXE_DTD, encoding="utf-8")
        with pytest.raises(ValueError, match="DTD and entity declarations"):
            parse_xml(malicious)
        with pytest.raises(ValueError, match="DTD and entity declarations"):
            parse_cpd(malicious)

        valid_file = tmp_path / "valid.xml"
        valid_file.write_text(VALID_PMD_XML, encoding="utf-8")
        symlink_file = tmp_path / "symlink.xml"
        symlink_file.symlink_to(valid_file)
        with pytest.raises(ValueError, match="symbolic link"):
            parse_xml(symlink_file)
        with pytest.raises(ValueError, match="symbolic link"):
            parse_cpd(symlink_file)
