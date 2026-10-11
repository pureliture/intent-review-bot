"""Adversarial stress-test suite for strict PMD and CPD XML parsers.

Targets: src/review_bot/adapters/pmd/parsers.py

Categories:
1. XXE & Entity Injection Attacks (SYSTEM, PUBLIC, Billion Laughs, parameter entities, multi-encoding DTDs)
2. Payload Boundary & Size Attacks (64 MiB boundaries, 0-byte payloads, binary garbage, malformed tokens)
3. Filesystem & Special Device Attacks (symlinks, dangling symlinks, FIFOs, sockets, device nodes)
4. Path Traversal & Source Path Sanitization (dotdot traversal, UNC, Windows paths, control chars, null bytes)
5. CPD & PMD Coordinate Boundary Quirks (single occurrence, duplicate coordinates, inverted spans, priorities)
6. Scale, Hash Determinism & Immutability
"""
from __future__ import annotations

import os
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest

from review_bot.adapters.pmd.parsers import (
    MAX_STRICT_REPORT_BYTES,
    _read_and_validate_payload,
    _source_path,
    parse_cpd_strict,
    parse_cpd_xml,
    parse_pmd_xml,
    parse_xml_strict,
)

# ==============================================================================
# Category 1: XXE & Entity Injection Attacks
# ==============================================================================


class TestXxeAndEntityInjectionAttacks:
    """Stress tests proving complete immunity against XXE and entity expansion attacks."""

    def test_system_external_entity_file_scheme_rejected(self) -> None:
        """Reject XML containing SYSTEM entity referencing local files."""
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE test [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">&xxe;</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_system_external_entity_http_scheme_rejected(self) -> None:
        """Reject XML containing SYSTEM entity referencing remote URLs."""
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE test [
  <!ENTITY xxe SYSTEM "http://127.0.0.1:8080/evil.dtd">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">&xxe;</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_public_external_entity_rejected(self) -> None:
        """Reject XML containing PUBLIC entity declarations."""
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE test PUBLIC "-//OASIS//DTD DocBook V4.1//EN" "http://example.com/docbook.dtd">
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_billion_laughs_recursive_entity_expansion_rejected(self) -> None:
        """Reject exponential Billion Laughs expansion bomb."""
        payload = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ELEMENT lolz (#PCDATA)>
 <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
 <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">
 <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
 <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">&lol4;</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_quadratic_entity_blowup_rejected(self) -> None:
        """Reject linear quadratic blowup attacks inside DTD."""
        huge_str = "a" * 50000
        payload = f"""<?xml version="1.0"?>
<!DOCTYPE bomb [
 <!ENTITY b "{huge_str}">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">&b;&b;&b;</violation>
  </file>
</pmd>
""".encode()
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_parameter_entities_rejected(self) -> None:
        """Reject parameter entity (%param;) declarations."""
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE test [
  <!ENTITY % dtd SYSTEM "http://127.0.0.1:8080/evil.dtd">
  %dtd;
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    @pytest.mark.parametrize(
        ("encoding", "bom"),
        [
            ("utf-8", b""),
            ("utf-8-sig", b"\xef\xbb\xbf"),
            ("utf-16", b""),
            ("utf-16le", b"\xff\xfe"),
            ("utf-16be", b"\xfe\xff"),
        ],
    )
    def test_xxe_across_encodings_with_bom(self, encoding: str, bom: bytes) -> None:
        """Verify DTD rejection operates across UTF-8, UTF-16, UTF-16LE, and UTF-16BE encodings."""
        xml_template = """<?xml version="1.0" encoding="UTF-16"?>
<!DOCTYPE test [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">&xxe;</violation>
  </file>
</pmd>
"""
        if "utf-8" in encoding:
            raw_xml = xml_template.replace('encoding="UTF-16"', 'encoding="UTF-8"')
            data = bom + raw_xml.encode("utf-8")
        else:
            raw_xml = xml_template
            data = bom + raw_xml.encode(encoding)

        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(data)

    def test_obfuscated_dtd_formatting_rejected(self) -> None:
        """Reject DTDs formatted with multiline whitespace, tabs, and comments."""
        payload = b"""<?xml version="1.0"?>
<!DOCTYPE
   foo
   [
      <!-- comment inside DTD -->
      <!ENTITY
         bar
         "baz"
      >
   ]
>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">&bar;</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_doctype_without_entities_rejected(self) -> None:
        """Any DOCTYPE header, even without entities, must be strictly forbidden."""
        payload = b"""<?xml version="1.0"?>
<!DOCTYPE pmd>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_pmd_xml(payload)

    def test_cdata_tricks_cannot_inject_dtd(self) -> None:
        """CDATA containing DOCTYPE text within an element is treated as text, not parsed as DTD."""
        payload = b"""<?xml version="1.0"?>
<pmd version="7.3.0">
  <file name="src/main/java/demo/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">
      <![CDATA[<!DOCTYPE evil SYSTEM "file:///etc/passwd">]]>
    </violation>
  </file>
</pmd>
"""
        report = parse_pmd_xml(payload)
        assert len(report.violations) == 1
        assert "<!DOCTYPE evil" in report.violations[0].message

    def test_cpd_xxe_rejection(self) -> None:
        """CPD parser must also strictly reject DTD and entity declarations."""
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE cpd [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="5" tokens="50">
    <file path="src/main/java/demo/A.java" line="1"/>
    <file path="src/main/java/demo/B.java" line="1"/>
    <codefragment>&xxe;</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            parse_cpd_xml(payload)

    def test_strict_wrappers_encapsulate_xxe_safely(self) -> None:
        """Strict wrapper functions parse_xml_strict and parse_cpd_strict encapsulate XXE as invalid."""
        payload = b"""<?xml version="1.0"?>
<!DOCTYPE test [ <!ENTITY xxe "evil"> ]>
<pmd version="7.3.0"/>
"""
        pmd_rep = parse_xml_strict(payload)
        assert pmd_rep.parse_status == "invalid"
        assert len(pmd_rep.errors) == 1
        assert "DTD and entity declarations" in pmd_rep.errors[0].message

        cpd_payload = b"""<?xml version="1.0"?>
<!DOCTYPE test [ <!ENTITY xxe "evil"> ]>
<pmd-cpd pmdVersion="7.3.0"/>
"""
        cpd_rep = parse_cpd_strict(cpd_payload)
        assert cpd_rep.parse_status == "invalid"
        assert len(cpd_rep.errors) == 1
        assert "DTD and entity declarations" in cpd_rep.errors[0].message


# ==============================================================================
# Category 2: Payload Boundary & Size Attacks
# ==============================================================================


class TestPayloadBoundariesAndMalformedPayloads:
    """Stress tests covering 64 MiB limit boundary, 0-byte files, and invalid tokens."""

    def test_exact_64mib_boundary_accepted_in_memory(self) -> None:
        """Payload of exactly 64 MiB (67,108,864 bytes) must be accepted when valid."""
        prefix = b'<pmd version="7.3.0"><file name="src/main/java/Foo.java"><violation beginline="1" endline="2" begincolumn="1" endcolumn="1" rule="R" ruleset="S" priority="1">msg</violation></file>'
        suffix = b"</pmd>"
        padding_len = MAX_STRICT_REPORT_BYTES - len(prefix) - len(suffix)
        exact_64mib = prefix + (b" " * padding_len) + suffix
        assert len(exact_64mib) == MAX_STRICT_REPORT_BYTES

        report = parse_pmd_xml(exact_64mib)
        assert report.parse_status == "valid"
        assert len(report.violations) == 1

    def test_exact_64mib_boundary_accepted_on_disk(self, tmp_path: Path) -> None:
        """On-disk file of exactly 64 MiB is accepted."""
        prefix = b'<pmd version="7.3.0"><file name="src/main/java/Foo.java"><violation beginline="1" endline="2" begincolumn="1" endcolumn="1" rule="R" ruleset="S" priority="1">msg</violation></file>'
        suffix = b"</pmd>"
        padding_len = MAX_STRICT_REPORT_BYTES - len(prefix) - len(suffix)
        exact_64mib = prefix + (b" " * padding_len) + suffix

        report_file = tmp_path / "64mib.xml"
        report_file.write_bytes(exact_64mib)

        report = parse_pmd_xml(report_file)
        assert report.parse_status == "valid"
        assert len(report.violations) == 1

    def test_64mib_plus_one_byte_rejected_in_memory(self) -> None:
        """Payload of 64 MiB + 1 byte (67,108,865 bytes) must be rejected with ValueError."""
        prefix = b"<pmd>"
        suffix = b"</pmd>"
        padding_len = (MAX_STRICT_REPORT_BYTES + 1) - len(prefix) - len(suffix)
        over_boundary = prefix + (b" " * padding_len) + suffix
        assert len(over_boundary) == MAX_STRICT_REPORT_BYTES + 1

        with pytest.raises(ValueError, match="Report exceeds the strict parser size limit"):
            parse_pmd_xml(over_boundary)

        with pytest.raises(ValueError, match="Report exceeds the strict parser size limit"):
            parse_cpd_xml(over_boundary)

    def test_64mib_plus_one_byte_rejected_on_disk(self, tmp_path: Path) -> None:
        """On-disk file of 64 MiB + 1 byte is rejected before parsing."""
        over_file = tmp_path / "over64mib.xml"
        with over_file.open("wb") as f:
            f.seek(MAX_STRICT_REPORT_BYTES)
            f.write(b"x")

        assert over_file.stat().st_size == MAX_STRICT_REPORT_BYTES + 1

        with pytest.raises(ValueError, match="Report exceeds the strict parser size limit"):
            parse_pmd_xml(over_file)

    def test_100mib_oversized_payload_rejected_immediately(self) -> None:
        """100 MiB bytearray rejected without memory exhaustion."""
        large = bytearray(100 * 1024 * 1024)
        with pytest.raises(ValueError, match="Report exceeds the strict parser size limit"):
            _read_and_validate_payload(bytes(large[:1000]) + bytes(large))

    def test_zero_byte_empty_input_rejected(self) -> None:
        """0-byte payload raises ET.ParseError or invalid status."""
        with pytest.raises(ET.ParseError):
            parse_pmd_xml(b"")

        with pytest.raises(ET.ParseError):
            parse_cpd_xml(b"")

        rep = parse_xml_strict(b"")
        assert rep.parse_status == "invalid"
        assert len(rep.errors) >= 1

    def test_zero_byte_file_on_disk_rejected(self, tmp_path: Path) -> None:
        """0-byte file on disk raises ET.ParseError."""
        empty_file = tmp_path / "empty.xml"
        empty_file.write_bytes(b"")

        with pytest.raises(ET.ParseError):
            parse_pmd_xml(empty_file)

    def test_whitespace_only_payload_rejected(self) -> None:
        """Whitespace-only payload raises ParseError."""
        with pytest.raises(ET.ParseError):
            parse_pmd_xml(b"   \n\t  \r\n  ")

    def test_binary_garbage_rejected(self) -> None:
        """Random non-XML binary garbage rejected."""
        garbage = b"\x00\xff\xfe\x01\x02\x03\x04\x80\x90\xaa"
        with pytest.raises(ET.ParseError):
            parse_pmd_xml(garbage)

    def test_truncated_xml_tokens_rejected(self) -> None:
        """Incomplete or truncated XML tokens rejected."""
        truncated_cases = [
            b"<pmd version='7.3.0'><file name='Foo.java'>",
            b"<pmd version='7.3.0'><file name='Foo.java'><violation>",
            b"<pmd-cpd pmdVersion='7.3.0'><duplication>",
            b"<?xml version='1.0'?>",
            b"<unclosed",
        ]
        for truncated in truncated_cases:
            with pytest.raises(ET.ParseError):
                parse_pmd_xml(truncated)

    def test_unsupported_root_element_tag_rejected(self) -> None:
        """Unrecognized root element tag rejected."""
        bad_roots = [
            b"<unknown version='7.3.0'/>",
            b"<checkstyle version='7.3.0'/>",
            b"<spotbugs version='7.3.0'/>",
        ]
        for bad in bad_roots:
            with pytest.raises(ValueError, match="Expected PMD report root and namespace"):
                parse_pmd_xml(bad)
            with pytest.raises(ValueError, match="Expected CPD report root and namespace"):
                parse_cpd_xml(bad)

    def test_missing_version_attribute_rejected(self) -> None:
        """Missing version attribute on root rejected."""
        with pytest.raises(ValueError, match="Missing report attribute: version"):
            parse_pmd_xml(b"<pmd/>")

        with pytest.raises(ValueError, match="Missing report attribute: pmdVersion"):
            parse_cpd_xml(b"<pmd-cpd/>")

    def test_forbidden_root_attributes_rejected(self) -> None:
        """Extra unauthorized attributes on root rejected."""
        with pytest.raises(ValueError, match="Unexpected report attribute or namespace"):
            parse_pmd_xml(b"<pmd version='7.3.0' malicious='true'/>")

        with pytest.raises(ValueError, match="Unexpected report attribute or namespace"):
            parse_cpd_xml(b"<pmd-cpd pmdVersion='7.3.0' evil='1'/>")


# ==============================================================================
# Category 3: Filesystem, Symlink & Device Node Attacks
# ==============================================================================


class TestFilesystemAndSpecialDeviceAttacks:
    """Stress tests verifying rejection of symlinks, FIFOs, sockets, and character devices."""

    def test_symlink_to_regular_file_rejected(self, tmp_path: Path) -> None:
        """Symbolic link to a valid regular file must be rejected."""
        real = tmp_path / "valid.xml"
        real.write_text('<pmd version="7.3.0"/>', encoding="utf-8")
        link = tmp_path / "link.xml"
        link.symlink_to(real)

        with pytest.raises(ValueError, match="Report must not be a symbolic link"):
            parse_pmd_xml(link)

        with pytest.raises(ValueError, match="Report must not be a symbolic link"):
            parse_cpd_xml(link)

    def test_dangling_symlink_rejected(self, tmp_path: Path) -> None:
        """Dangling symbolic link pointing to non-existent target must be rejected."""
        dangling = tmp_path / "dangling.xml"
        dangling.symlink_to(tmp_path / "nonexistent.xml")

        with pytest.raises(ValueError, match="Report must not be a symbolic link"):
            parse_pmd_xml(dangling)

    def test_fifo_named_pipe_rejected(self, tmp_path: Path) -> None:
        """FIFO (named pipe) rejected before opening/reading."""
        fifo_path = tmp_path / "attack.fifo"
        os.mkfifo(fifo_path)

        with pytest.raises(ValueError, match="Report must be a regular file"):
            parse_pmd_xml(fifo_path)

        with pytest.raises(ValueError, match="Report must be a regular file"):
            parse_cpd_xml(fifo_path)

    def test_unix_domain_socket_rejected(self) -> None:
        """UNIX domain socket rejected before opening."""
        # Use short path in /tmp to satisfy Darwin 104-byte sockaddr limit
        sock_path = Path(f"/tmp/rb_test_{os.getpid()}.sock")
        if sock_path.exists():
            sock_path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(sock_path))
        try:
            with pytest.raises(ValueError, match="Report must be a regular file"):
                parse_pmd_xml(sock_path)
        finally:
            sock.close()
            sock_path.unlink(missing_ok=True)

    def test_directory_path_rejected(self, tmp_path: Path) -> None:
        """Directory passed as report file rejected."""
        dir_path = tmp_path / "directory_not_file"
        dir_path.mkdir()

        with pytest.raises(ValueError, match="Report must be a regular file"):
            parse_pmd_xml(dir_path)

    def test_character_device_dev_null_rejected(self) -> None:
        """Character device nodes like /dev/null rejected as non-regular files."""
        dev_null = Path("/dev/null")
        if dev_null.exists():
            with pytest.raises(ValueError, match="Report must be a regular file"):
                parse_pmd_xml(dev_null)

    def test_nonexistent_path_behavior(self, tmp_path: Path) -> None:
        """Non-existent Path raises FileNotFoundError when raise_on_error=True."""
        missing = tmp_path / "missing_file.xml"
        with pytest.raises(FileNotFoundError):
            parse_pmd_xml(missing)

        rep = parse_pmd_xml(missing, raise_on_error=False)
        assert rep.exists is False
        assert rep.parse_status == "invalid"

    def test_unsupported_payload_types_raise_type_error(self) -> None:
        """Passing int, list, or dict as report payload raises TypeError."""
        for invalid in [123, ["<pmd/>"], {"xml": "data"}]:
            with pytest.raises(TypeError, match="Unsupported report payload type"):
                parse_pmd_xml(invalid)  # type: ignore[arg-type]


# ==============================================================================
# Category 4: Path Traversal & Source Path Sanitization
# ==============================================================================


class TestPathTraversalAndSanitization:
    """Stress tests covering directory traversal, Windows separators, UNC, control chars, and snapshot escapes."""

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../etc/passwd",
            "../../secret.key",
            "foo/../../bar.java",
            "src/../../../etc/shadow",
            "C:\\Windows\\System32\\calc.exe",
            "D:\\repo\\File.java",
            "foo\\bar.java",
            "//remote/share/Foo.java",
            "\\\\network\\share\\Foo.java",
            "~/secret.java",
            "~root/file.java",
            "foo:bar.java",
            "C:file.java",
            "foo\x00bar.java",
            "foo\nbar.java",
            "foo\rbar.java",
            "foo\tbar.java",
            "foo\x7fbar.java",
            "foo\x1fbar.java",
            "",
        ],
    )
    def test_source_path_direct_sanitization(self, bad_path: str) -> None:
        """_source_path directly rejects all unsafe path inputs."""
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            _source_path(bad_path, source_root=None)

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../etc/passwd",
            "../../secret.key",
            "foo/../../bar.java",
            "src/../../../etc/shadow",
            "C:\\Windows\\System32\\calc.exe",
            "D:\\repo\\File.java",
            "foo\\bar.java",
            "//remote/share/Foo.java",
            "~/secret.java",
            "foo:bar.java",
            "C:file.java",
            "foo\nbar.java",
            "foo\rbar.java",
            "foo\tbar.java",
            "foo\x7fbar.java",
        ],
    )
    def test_unsafe_paths_rejected_in_pmd_and_cpd_xml(self, bad_path: str) -> None:
        """Unsafe paths embedded in PMD and CPD XML reports are rejected."""
        pmd_xml = f"""<pmd version="7.3.0">
  <file name={quoteattr(bad_path)}>
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            parse_pmd_xml(pmd_xml)

        cpd_xml = f"""<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="5" tokens="50">
    <file path={quoteattr(bad_path)} line="1"/>
    <file path="src/main/java/valid.java" line="1"/>
    <codefragment>snippet</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            parse_cpd_xml(cpd_xml)

    @pytest.mark.parametrize("empty_or_ws", ["", "   ", " \t "])
    def test_empty_or_whitespace_filename_rejected(self, empty_or_ws: str) -> None:
        """Empty or whitespace-only file names rejected by parser as missing attribute."""
        pmd_xml = f"""<pmd version="7.3.0">
  <file name={quoteattr(empty_or_ws)}>
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Missing report attribute: name"):
            parse_pmd_xml(pmd_xml)

    def test_snapshot_escape_when_source_root_is_provided(self, tmp_path: Path) -> None:
        """Paths pointing outside source_root must be strictly rejected."""
        source_root = tmp_path / "workspace"
        source_root.mkdir()
        outside_file = tmp_path / "outside.java"
        outside_file.write_text("class Outside {}", encoding="utf-8")

        xml = f"""<pmd version="7.3.0">
  <file name="{outside_file.as_posix()}">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references code outside its revision snapshot"):
            parse_pmd_xml(xml, source_root=source_root)

    def test_symlink_escape_within_source_root_rejected(self, tmp_path: Path) -> None:
        """Symlinks within source_root pointing to outside targets must be rejected."""
        source_root = tmp_path / "workspace"
        source_root.mkdir()
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret", encoding="utf-8")

        symlink_in_root = source_root / "Escaped.java"
        symlink_in_root.symlink_to(outside_file)

        xml = """<pmd version="7.3.0">
  <file name="Escaped.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references code outside its revision snapshot"):
            parse_pmd_xml(xml, source_root=source_root)

    def test_error_and_suppressed_filename_sanitization(self) -> None:
        """Filename in <error> and <suppressedviolation> must also undergo path sanitization."""
        xml_err = """<pmd version="7.3.0">
  <error filename="../traversal.java" msg="Error msg">details</error>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            parse_pmd_xml(xml_err)

        xml_sup = """<pmd version="7.3.0">
  <suppressedviolation filename="../traversal.java"
                       suppressiontype="annotation"
                       msg="Suppressed"
                       usermsg="User reason"/>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            parse_pmd_xml(xml_sup)

    def test_source_contents_dict_strict_membership(self) -> None:
        """Referencing file absent from source_contents dictionary raises ValueError."""
        contents = {"src/main/java/Present.java": "public class Present {}"}
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Absent.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report references code absent from the fixed source data"):
            parse_pmd_xml(xml, source_contents=contents)

    def test_span_exceeds_source_contents_line_count(self) -> None:
        """Violation span exceeding line count of source_contents raises ValueError."""
        contents = {"src/main/java/Short.java": "line 1\nline 2\n"}
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Short.java">
    <violation beginline="1" endline="10" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Report span exceeds the source file"):
            parse_pmd_xml(xml, source_contents=contents)


# ==============================================================================
# Category 5: CPD & PMD Coordinate Boundary Quirks
# ==============================================================================


class TestCoordinateAndStructuralBoundaryQuirks:
    """Stress tests covering CPD occurrence counts, duplicate locations, PMD coordinates and priorities."""

    def test_cpd_group_with_zero_occurrences_rejected(self) -> None:
        """CPD group with 0 file occurrences rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group needs at least two locations"):
            parse_cpd_xml(xml)

    def test_cpd_group_with_single_occurrence_rejected(self) -> None:
        """CPD group with only 1 file occurrence must be rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/Single.java" line="5"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group needs at least two locations and one code fragment"):
            parse_cpd_xml(xml)

    def test_cpd_group_with_duplicate_locations_rejected(self) -> None:
        """CPD group with duplicate coordinates rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/Duplicate.java" line="10" column="5"/>
    <file path="src/main/java/Duplicate.java" line="10" column="5"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group repeats the same source location"):
            parse_cpd_xml(xml)

    def test_cpd_group_with_duplicate_locations_without_columns_rejected(self) -> None:
        """CPD group with duplicate coordinates when column is omitted rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/Duplicate.java" line="10"/>
    <file path="src/main/java/Duplicate.java" line="10"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group repeats the same source location"):
            parse_cpd_xml(xml)

    def test_cpd_group_with_zero_or_multiple_codefragments_rejected(self) -> None:
        """CPD group must have exactly one codefragment."""
        xml0 = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/A.java" line="1"/>
    <file path="src/main/java/B.java" line="1"/>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group needs at least two locations and one code fragment"):
            parse_cpd_xml(xml0)

        xml2 = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/A.java" line="1"/>
    <file path="src/main/java/B.java" line="1"/>
    <codefragment>f1</codefragment>
    <codefragment>f2</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group needs at least two locations and one code fragment"):
            parse_cpd_xml(xml2)

    def test_cpd_group_with_empty_codefragment_rejected(self) -> None:
        """CPD group with empty codefragment text rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/A.java" line="1"/>
    <file path="src/main/java/B.java" line="1"/>
    <codefragment>   \n   </codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="CPD group needs at least two locations and one code fragment"):
            parse_cpd_xml(xml)

    def test_cpd_inverted_line_span_rejected(self) -> None:
        """CPD occurrence with endline < line rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="10" tokens="100">
    <file path="src/main/java/A.java" line="20" endline="10"/>
    <file path="src/main/java/B.java" line="5" endline="15"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="Invalid CPD source span"):
            parse_cpd_xml(xml)

    def test_cpd_inverted_column_span_on_same_line_rejected(self) -> None:
        """CPD occurrence with line == endline and endcolumn < begincolumn rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="1" tokens="10">
    <file path="src/main/java/A.java" line="5" endline="5" column="20" endcolumn="10"/>
    <file path="src/main/java/B.java" line="5" endline="5" column="5" endcolumn="15"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="Invalid CPD source span"):
            parse_cpd_xml(xml)

    def test_cpd_conflicting_begin_columns_rejected(self) -> None:
        """CPD occurrence specifying conflicting column and begincolumn rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="5" tokens="50">
    <file path="src/main/java/A.java" line="1" column="5" begincolumn="10"/>
    <file path="src/main/java/B.java" line="1"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="Conflicting CPD begin columns"):
            parse_cpd_xml(xml)

    def test_cpd_inverted_token_span_rejected(self) -> None:
        """CPD occurrence specifying endtoken < begintoken rejected."""
        xml = """<pmd-cpd pmdVersion="7.3.0">
  <duplication lines="5" tokens="50">
    <file path="src/main/java/A.java" line="1" begintoken="100" endtoken="50"/>
    <file path="src/main/java/B.java" line="1"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match="Invalid CPD token span"):
            parse_cpd_xml(xml)

    @pytest.mark.parametrize(
        ("attr", "val"),
        [
            ("lines", "0"),
            ("lines", "-5"),
            ("lines", "abc"),
            ("tokens", "0"),
            ("tokens", "-10"),
            ("tokens", "xyz"),
        ],
    )
    def test_cpd_invalid_metric_attributes_rejected(self, attr: str, val: str) -> None:
        """CPD lines and tokens must be positive integers."""
        lines_attr = f'lines="{val}"' if attr == "lines" else 'lines="10"'
        tokens_attr = f'tokens="{val}"' if attr == "tokens" else 'tokens="100"'
        xml = f"""<pmd-cpd pmdVersion="7.3.0">
  <duplication {lines_attr} {tokens_attr}>
    <file path="src/main/java/A.java" line="1"/>
    <file path="src/main/java/B.java" line="1"/>
    <codefragment>code</codefragment>
  </duplication>
</pmd-cpd>
"""
        with pytest.raises(ValueError, match=r"Invalid or missing (nonnegative|positive) integer"):
            parse_cpd_xml(xml)

    def test_pmd_file_with_zero_violations_rejected(self) -> None:
        """PMD <file> element with 0 violations rejected."""
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Clean.java"/>
</pmd>
"""
        with pytest.raises(ValueError, match="PMD file needs at least one violation"):
            parse_pmd_xml(xml)

    def test_pmd_violation_with_empty_message_rejected(self) -> None:
        """PMD violation with empty or whitespace message rejected."""
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">
       \n   \t  
    </violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Missing PMD violation message"):
            parse_pmd_xml(xml)

    def test_pmd_inverted_line_span_rejected(self) -> None:
        """PMD violation with beginline > endline rejected."""
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="50" endline="25" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Invalid PMD source span or priority"):
            parse_pmd_xml(xml)

    def test_pmd_inverted_column_span_on_same_line_rejected(self) -> None:
        """PMD violation with beginline == endline and begincolumn > endcolumn rejected."""
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="10" endline="10" begincolumn="25" endcolumn="5"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError, match="Invalid PMD source span or priority"):
            parse_pmd_xml(xml)

    def test_pmd_single_character_column_span_accepted(self) -> None:
        """PMD violation with begincolumn == endcolumn on same line is accepted."""
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="10" endline="10" begincolumn="5" endcolumn="5"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        report = parse_pmd_xml(xml)
        assert len(report.violations) == 1
        assert report.violations[0].begin_column == 5
        assert report.violations[0].end_column == 5

    def test_pmd_multiline_span_with_smaller_endcolumn_accepted(self) -> None:
        """In multi-line spans, endcolumn may be smaller than begincolumn."""
        xml = """<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="10" endline="12" begincolumn="40" endcolumn="10"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        report = parse_pmd_xml(xml)
        assert len(report.violations) == 1
        assert report.violations[0].begin_line == 10
        assert report.violations[0].end_line == 12
        assert report.violations[0].begin_column == 40
        assert report.violations[0].end_column == 10

    @pytest.mark.parametrize(
        "bad_priority",
        ["0", "6", "7", "-1", "999", "high", "low", "1.5"],
    )
    def test_pmd_priority_outside_valid_range_rejected(self, bad_priority: str) -> None:
        """PMD priority outside 1..5 rejected."""
        xml = f"""<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="{bad_priority}">msg</violation>
  </file>
</pmd>
"""
        with pytest.raises(ValueError):
            parse_pmd_xml(xml)

    @pytest.mark.parametrize("good_priority", [1, 2, 3, 4, 5])
    def test_pmd_valid_priorities_1_to_5_accepted(self, good_priority: int) -> None:
        """PMD priorities 1..5 all accepted."""
        xml = f"""<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="{good_priority}">msg</violation>
  </file>
</pmd>
"""
        report = parse_pmd_xml(xml)
        assert report.violations[0].priority == good_priority


# ==============================================================================
# Category 6: Scale, Memory & Hash Determinism
# ==============================================================================


class TestScaleMemoryAndIntegrity:
    """Stress tests verifying throughput on scaled reports, SHA-256 hash determinism, and immutability."""

    def test_large_pmd_report_1000_violations_throughput(self) -> None:
        """Report with 1,000 PMD violations parses in < 1 second."""
        violations_xml = []
        for i in range(1, 1001):
            violations_xml.append(
                f'<violation beginline="{i}" endline="{i+1}" begincolumn="1" endcolumn="10" '
                f'rule="Rule{i % 10}" ruleset="Design" priority="{(i % 5) + 1}">Violation message {i}</violation>'
            )
        all_violations = "\n".join(violations_xml)
        xml = f"""<pmd version="7.3.0">
  <file name="src/main/java/BigFile.java">
    {all_violations}
  </file>
</pmd>
"""
        t0 = time.perf_counter()
        report = parse_pmd_xml(xml)
        duration = time.perf_counter() - t0

        assert len(report.violations) == 1000
        assert report.finding_count == 1000
        assert duration < 1.0, f"Parsing 1,000 violations took {duration:.2f}s (too slow)"

    def test_large_cpd_report_100_duplications_throughput(self) -> None:
        """Report with 100 CPD duplications parses in < 1 second."""
        dups_xml = []
        for i in range(1, 101):
            dups_xml.append(
                f'<duplication lines="20" tokens="150">'
                f'<file path="src/main/java/FileA{i}.java" line="10"/>'
                f'<file path="src/main/java/FileB{i}.java" line="30"/>'
                f'<codefragment>duplicate snippet {i}</codefragment>'
                f'</duplication>'
            )
        all_dups = "\n".join(dups_xml)
        xml = f"""<pmd-cpd pmdVersion="7.3.0">
  {all_dups}
</pmd-cpd>
"""
        t0 = time.perf_counter()
        report = parse_cpd_xml(xml)
        duration = time.perf_counter() - t0

        assert len(report.duplications) == 100
        assert report.finding_count == 100
        assert duration < 1.0, f"Parsing 100 duplications took {duration:.2f}s (too slow)"

    def test_sha256_content_hash_determinism(self) -> None:
        """Identical XML payload produces identical content_hash across multiple calls."""
        xml = b"""<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        rep1 = parse_pmd_xml(xml)
        rep2 = parse_pmd_xml(xml)
        assert rep1.content_hash == rep2.content_hash
        assert len(rep1.content_hash) == 64

    def test_parsed_report_immutability(self) -> None:
        """PmdReport and CpdReport are frozen dataclasses; mutations must raise FrozenInstanceError."""
        xml = b"""<pmd version="7.3.0">
  <file name="src/main/java/Foo.java">
    <violation beginline="1" endline="2" begincolumn="1" endcolumn="1"
               rule="R" ruleset="S" priority="1">msg</violation>
  </file>
</pmd>
"""
        report = parse_pmd_xml(xml)
        with pytest.raises(FrozenInstanceError):
            report.violations = ()  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            report.tool = "other"  # type: ignore[misc]
