from __future__ import annotations

from dataclasses import replace

import pytest

from review_bot.application.source_partition import (
    PartitionError, SourceOriginal, partition_original, verify_partition,
)


@pytest.mark.parametrize("path,text", [
    ("a.py", "# intro\n" + "\n".join(f"def f{i}():\n    return {i}\n" for i in range(12))),
    ("A.java", "class A {\n" + "\n".join(f" int f{i}() {{ return {i}; }}" for i in range(12)) + "\n}\n"),
    ("schema.sql", "-- source\r\n" + "CREATE TABLE t(x TEXT);\r\n" * 20),
    ("broken.py", "def f(:\n" + "    return 1\n" * 20),
    ("Broken.java", "class A { void f( {\n" + "int i = 1;\n" * 20),
    ("long.py", "def f():\n    return '" + "한😀" * 200 + "'\n"),
])
def test_independent_ast_and_fallback_preserve_exact_original(path, text):
    original = SourceOriginal("E1", text, "source", path, "head", "a" * 40)
    fits = lambda span: len(span.text.encode()) <= 80
    spans = partition_original(original, fits)
    assert len(spans) > 1
    verify_partition(original, spans)
    assert b"".join(s.text.encode() for s in spans) == text.encode()
    assert all(len(s.text.encode()) <= 80 for s in spans)
    assert spans == partition_original(original, fits)


def test_fitting_functions_remain_whole():
    text = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(8))
    original = SourceOriginal("E", text, "source", "a.py")
    spans = partition_original(original, lambda s: len(s.text.encode()) <= 55)
    assert all("def f" not in s.text or s.text.count("def f") == s.text.count("return") for s in spans)
    assert any(s.boundary == "syntax_boundary" for s in spans)


def test_empty_source_and_unusable_request_budget_are_explicit():
    original = SourceOriginal("E", "", "source", "a.py")
    verify_partition(original, partition_original(original, lambda _: True))
    with pytest.raises(PartitionError, match="fixed_request_exceeds"):
        partition_original(original, lambda _: False)


def test_tampered_offsets_and_content_are_rejected():
    original = SourceOriginal("E", "def f():\n    return 1\n", "source", "a.py")
    spans = partition_original(original, lambda _: True)
    with pytest.raises(PartitionError):
        verify_partition(original, (replace(spans[0], text="wrong"),))


def test_measurement_failure_is_not_reinterpreted_as_a_byte_limit():
    def unavailable(_):
        raise RuntimeError("native measurement unavailable")
    with pytest.raises(RuntimeError, match="native measurement"):
        partition_original(SourceOriginal("E", "def f(): pass", "source", "a.py"), unavailable)
