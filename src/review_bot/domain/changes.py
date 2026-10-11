"""Six change states and measurement types for delta analysis."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ChangeState(str, Enum):
    """Six deterministic static analysis change states."""

    EXISTING = "existing"
    NEW = "new"
    WORSENED = "worsened"
    IMPROVED = "improved"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


Change = Literal["existing", "new", "worsened", "improved", "resolved", "unknown"]

CHANGE_STATES: tuple[Change, ...] = (
    "existing",
    "new",
    "worsened",
    "improved",
    "resolved",
    "unknown",
)


@dataclass(frozen=True)
class Measurement:
    """Quantitative measurement before and after change."""

    name: str
    base: float | int | None
    head: float | int | None
