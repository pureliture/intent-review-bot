"""Automated AST and runtime verification of domain purity."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import review_bot.domain

ALLOWED_MODULES = {"__future__", "dataclasses", "enum", "typing", "review_bot.domain"}
FORBIDDEN_KEYWORDS = {
    "os",
    "sys",
    "pathlib",
    "io",
    "shutil",
    "socket",
    "http",
    "urllib",
    "requests",
    "httpx",
    "sqlite3",
    "pydantic",
    "fastapi",
    "yaml",
    "subprocess",
    "threading",
    "multiprocessing",
    "asyncio",
}


def test_ast_purity() -> None:
    domain_path = Path(review_bot.domain.__file__).parent
    py_files = list(domain_path.glob("*.py"))
    assert len(py_files) >= 5, "Expected at least 5 domain python files"

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert (
                        root_mod in ALLOWED_MODULES
                    ), f"Forbidden import '{alias.name}' in {py_file.name}"
                    assert (
                        root_mod not in FORBIDDEN_KEYWORDS
                    ), f"IO/Forbidden import '{root_mod}' in {py_file.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    continue  # Intra-package relative import is permitted
                if node.module:
                    root_mod = node.module.split(".")[0]
                    assert (
                        root_mod in ALLOWED_MODULES
                    ), f"Forbidden import from '{node.module}' in {py_file.name}"
                    assert (
                        root_mod not in FORBIDDEN_KEYWORDS
                    ), f"IO/Forbidden import from '{root_mod}' in {py_file.name}"


def test_dataclasses_are_frozen() -> None:
    dataclass_count = 0
    from enum import Enum

    for name, obj in inspect.getmembers(review_bot.domain, inspect.isclass):
        if getattr(obj, "__module__", "").startswith("review_bot.domain"):
            if issubclass(obj, Enum):
                continue
            params = getattr(obj, "__dataclass_params__", None)
            assert (
                params is not None
            ), f"Class {name} in review_bot.domain must be a dataclass"
            assert params.frozen is True, f"Class {name} must be a frozen dataclass"
            dataclass_count += 1
    assert dataclass_count >= 15, f"Expected at least 15 frozen dataclasses, got {dataclass_count}"
