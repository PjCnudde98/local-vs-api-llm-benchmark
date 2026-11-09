"""Load variant, cost and rubric configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import ROOT

load_dotenv(ROOT / ".env")


@dataclass
class Variant:
    key: str
    label: str
    provider: str  # azure | openai_compatible | fake
    model: str
    cloud: bool = False
    pii: str = "none"  # none | raw | pseudonymise
    allow_real_data: bool = False
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 800
    token_param: str = "max_tokens"
    json_mode: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _expand(value: Any) -> Any:
    return os.path.expandvars(value) if isinstance(value, str) else value


def _build(key: str, raw: dict[str, Any], defaults: dict[str, Any]) -> Variant:
    merged = {**defaults, **{k: _expand(v) for k, v in raw.items()}}
    known = Variant.__dataclass_fields__.keys() - {"key", "extra"}
    kwargs = {k: v for k, v in merged.items() if k in known}
    extra = {k: v for k, v in merged.items() if k not in known}
    return Variant(key=key, extra=extra, **kwargs)


def load_variants(path: Path | None = None) -> dict[str, Variant]:
    data = yaml.safe_load((path or ROOT / "config" / "variants.yaml").read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    out = {k: _build(k, v, defaults) for k, v in data["variants"].items()}
    out["JUDGE"] = _build("JUDGE", data["judge"], defaults)
    return out


def load_costs(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or ROOT / "config" / "costs.yaml").read_text(encoding="utf-8"))


def load_rubric(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or ROOT / "config" / "rubric.yaml").read_text(encoding="utf-8"))


def load_gates(path: Path | None = None) -> dict[str, Any]:
    """Decision thresholds (machine-readable copy of docs/gates.md)."""
    return yaml.safe_load((path or ROOT / "config" / "gates.yaml").read_text(encoding="utf-8"))
