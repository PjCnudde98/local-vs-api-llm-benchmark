"""Load the lineup (models x routes x reasoning modes), costs, rubric and gates."""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from . import ROOT

load_dotenv(ROOT / ".env")

SIZE_CLASSES = ("cloud", "xl", "large", "medium", "small")


@dataclass
class Variant:
    key: str
    label: str
    provider: str  # azure | openai_compatible | anthropic | fake | hybrid
    model: str
    cloud: bool = False
    pii: str = "none"  # none | raw | pseudonymise | abstract
    allow_real_data: bool = False
    base_url: str | None = None
    temperature: float | None = 0.2  # None = not sent (Claude API, reasoning GPT models)
    max_tokens: int = 800  # used when a tier has no budget of its own
    max_tokens_by_tier: dict[int, int] = field(default_factory=dict)
    reasoning_extra_tokens: int = 0
    token_param: str = "max_tokens"
    json_mode: bool = True
    # Lineup metadata (set when the variant is expanded from `models` x `routes`).
    model_key: str = ""
    model_label: str = ""
    family: str = ""
    size_class: str = ""  # cloud | xl | large | medium | small | none
    route: str = ""  # local | pseudonymise | abstract | raw | own_tenant | hybrid
    reasoning: str = "default"  # default | on | off
    request: dict[str, Any] = field(default_factory=dict)  # per-mode request overrides
    hosting: str = ""
    hardware: str = ""  # cost profile in config/costs.yaml (local models)
    abstractor: str | None = None  # abstract route: local variant that writes the summary
    hybrid: dict[str, Any] = field(default_factory=dict)  # router, easy, hard, hard_tiers, router_tiers
    extra: dict[str, Any] = field(default_factory=dict)

    def tokens_for(self, tier: int | None) -> int:
        base = self.max_tokens_by_tier.get(tier, self.max_tokens) if tier else self.max_tokens
        return base + (self.reasoning_extra_tokens if self.reasoning == "on" else 0)


_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default}; an unset VAR without default stays as written."""
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m[1]) or (m[2] if m[2] is not None else m[0]), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _build(key: str, raw: dict[str, Any], defaults: dict[str, Any]) -> Variant:
    merged = {**defaults, **_expand(raw)}
    if merged.get("kind") == "hybrid":
        merged["hybrid"] = {k: merged.pop(k) for k in ("router", "easy", "hard", "hard_tiers", "router_tiers")
                            if k in merged}
        merged.update(provider="hybrid", model="", route="hybrid", size_class="hybrid")
        merged.pop("kind")
    known = Variant.__dataclass_fields__.keys() - {"key", "extra"}
    kwargs = {k: v for k, v in merged.items() if k in known}
    if "max_tokens_by_tier" in kwargs:
        kwargs["max_tokens_by_tier"] = {int(t): int(n) for t, n in kwargs["max_tokens_by_tier"].items()}
    extra = {k: v for k, v in merged.items() if k not in known}
    return Variant(key=key, extra=extra, **kwargs)


def variant_key(route: str, model_key: str, mode: str) -> str:
    return f"{route}/{model_key}" + ("+think" if mode == "on" else "")


def _expand_models(data: dict, defaults: dict) -> dict[str, Variant]:
    routes = data.get("routes") or {}
    out: dict[str, Variant] = {}
    for mkey, m in (data.get("models") or {}).items():
        m = dict(m)
        # YAML 1.1 reads the keys `on`/`off` as booleans: map them back.
        modes = {({True: "on", False: "off"}.get(k, k) if isinstance(k, bool) else str(k)): r
                 for k, r in (m.pop("reasoning", None) or {"default": {}}).items()}
        model_routes = m.pop("routes", None) or ["local"]
        label = m.pop("label", mkey)
        for route in model_routes:
            if route not in routes:
                raise ValueError(f"model {mkey}: unknown route {route!r}; have {sorted(routes)}")
            for mode, request in modes.items():
                key = variant_key(route, mkey, mode)
                raw = {**m, **routes[route], "model_key": mkey, "model_label": label, "route": route,
                       "reasoning": mode, "request": request or {},
                       "label": f"{label} / {m.get('hosting', '')} / {route}" + (" / thinking" if mode == "on" else "")}
                out[key] = _build(key, raw, defaults)
    return out


def load_variants(path: Path | None = None) -> dict[str, Variant]:
    """Every runnable variant plus the judges, keyed by variant key."""
    data = yaml.safe_load((path or ROOT / "config" / "variants.yaml").read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    out = _expand_models(data, defaults)
    for k, v in (data.get("variants") or {}).items():
        out[k] = _build(k, v, defaults)
    judges = data.get("judges") or ({"JUDGE": data["judge"]} if "judge" in data else {})
    for k, v in judges.items():
        out[k] = _build(k, v, defaults)
    return out


def is_judge(key: str) -> bool:
    return key.startswith("JUDGE")


def select(variants: dict[str, Variant], patterns: list[str]) -> list[str]:
    """Variant keys matching any glob pattern, in lineup order. Judges are never selected by a glob."""
    keys = [k for k in variants if not is_judge(k)]
    out: list[str] = []
    for p in patterns:
        hits = [k for k in keys if fnmatch.fnmatchcase(k, p)] if any(c in p for c in "*?[") else (
            [p] if p in variants else [])
        if not hits:
            raise KeyError(f"no variant matches {p!r}; run `cexp variants` for the list")
        out += [h for h in hits if h not in out]
    return out


def load_costs(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or ROOT / "config" / "costs.yaml").read_text(encoding="utf-8"))


def load_rubric(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or ROOT / "config" / "rubric.yaml").read_text(encoding="utf-8"))


def load_gates(path: Path | None = None) -> dict[str, Any]:
    """Decision thresholds (machine-readable copy of docs/gates.md)."""
    return yaml.safe_load((path or ROOT / "config" / "gates.yaml").read_text(encoding="utf-8"))
