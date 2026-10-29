"""Configurable publish gate for embodied-AI datasets.

The structural validator answers "is the file layout correct?". A gate adds
policy: the expected control frequency, the features training requires, the
minimum health score, and optional frame-quality thresholds. Rules can live in
JSON/TOML or be passed as a mapping by an API/MCP caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model import Issue, Severity, summarize_issues
from .reports import quality_report
from .validate import validate

ProgressCallback = Any

DEFAULT_RULES: dict[str, Any] = {
    "expected_fps": None,
    "fps_tolerance": 0.5,
    "fps_tolerance_ratio": 0.01,
    "required_features": [],
    "required_feature_kinds": [],
    "min_episodes": 1,
    "min_frames": None,
    "min_score": 80,
    "max_errors": 0,
    "max_warnings": None,
    "max_nan_ratio": None,
    "max_anomaly_windows": None,
    "max_error_anomalies": None,
    "max_jump_windows": None,
    "max_stall_windows": None,
    "checks": None,
    "include_frames": False,
    "use_cache": True,
    "chunk_frames": None,
    "max_frames": None,
}

_NUMBER_RULES = {
    "expected_fps",
    "fps_tolerance",
    "fps_tolerance_ratio",
    "min_score",
    "max_errors",
    "max_warnings",
    "max_nan_ratio",
    "max_anomaly_windows",
    "max_error_anomalies",
    "max_jump_windows",
    "max_stall_windows",
    "chunk_frames",
    "max_frames",
}


def _read_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ImportError as exc:  # pragma: no cover - Python 3.11+
            raise ValueError("TOML rules require Python 3.11+ or a JSON rules file") from exc
        payload = tomllib.loads(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            try:
                import tomllib
            except ImportError as exc:  # pragma: no cover
                raise ValueError(f"{path}: expected JSON rules") from exc
            payload = tomllib.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: rules document must be an object")
    if isinstance(payload.get("gate"), dict):
        payload = payload["gate"]
    return payload


def load_quality_rules(source: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load rules, merge them over defaults, and reject misspelled keys."""

    if source is None:
        return dict(DEFAULT_RULES)
    if isinstance(source, Mapping):
        provided = dict(source)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        provided = _read_document(path)
    unknown = sorted(set(provided) - set(DEFAULT_RULES))
    if unknown:
        raise ValueError(f"unknown quality gate rule(s): {unknown}; allowed: {sorted(DEFAULT_RULES)}")
    rules = dict(DEFAULT_RULES)
    rules.update(provided)
    for key in _NUMBER_RULES:
        if rules.get(key) is not None:
            try:
                rules[key] = float(rules[key]) if key.endswith("ratio") or key in {"expected_fps", "fps_tolerance"} else int(rules[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"rule {key!r} must be numeric") from exc
    for key in ("required_features", "required_feature_kinds", "checks"):
        if rules.get(key) is not None:
            if not isinstance(rules[key], (list, tuple)):
                raise ValueError(f"rule {key!r} must be a list")
            rules[key] = [str(value) for value in rules[key]]
    for key in ("include_frames", "use_cache"):
        rules[key] = bool(rules[key])
    return rules


def _gate_issue(code: str, message: str, *, hint: str | None = None, detail: dict[str, Any] | None = None) -> Issue:
    return Issue(
        code=code,
        severity=Severity.ERROR,
        message=message,
        scope="gate",
        hint=hint,
        detail=detail or {},
    )


def evaluate_quality_gate(
    path: str | Path,
    *,
    rules: str | Path | Mapping[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Validate a dataset against policy rules and return a JSON-ready report."""

    resolved = load_quality_rules(rules)

    def report(value: float, message: str) -> None:
        if progress is not None:
            progress(value, message)

    validation = validate(
        path,
        checks=resolved["checks"] or None,
        include_frames=bool(resolved["include_frames"]),
        progress=(lambda value, message: report(value * 0.62, message)) if progress is not None else None,
    )
    summary = validation.get("summary") or {}
    issues: list[Issue] = []
    failures: list[dict[str, Any]] = []

    def fail(code: str, message: str, *, hint: str | None = None, detail: dict[str, Any] | None = None) -> None:
        issue = _gate_issue(code, message, hint=hint, detail=detail)
        issues.append(issue)
        failures.append({"code": code, "message": message, "detail": detail or {}})

    errors = int(validation.get("errors", 0))
    warnings = int(validation.get("warnings", 0))
    if resolved["max_errors"] is not None and errors > int(resolved["max_errors"]):
        fail("gate.errors", f"{errors} error-level issues exceed the limit {int(resolved['max_errors'])}")
    if resolved["max_warnings"] is not None and warnings > int(resolved["max_warnings"]):
        fail("gate.warnings", f"{warnings} warnings exceed the limit {int(resolved['max_warnings'])}")
    if validation.get("score") is not None and float(validation["score"]) < float(resolved["min_score"]):
        fail("gate.score", f"health score {validation['score']} is below {resolved['min_score']}")

    episodes = int(summary.get("episodes") or 0)
    frames = summary.get("total_frames")
    if episodes < int(resolved["min_episodes"]):
        fail("gate.episodes", f"{episodes} episodes is below the required {resolved['min_episodes']}")
    if resolved["min_frames"] is not None and (frames is None or int(frames) < int(resolved["min_frames"])):
        fail("gate.frames", f"frame count {frames} is below the required {resolved['min_frames']}")

    expected_fps = resolved["expected_fps"]
    actual_fps = summary.get("fps")
    if expected_fps is not None:
        if actual_fps is None:
            fail("gate.fps_missing", f"expected {expected_fps} fps but the dataset declares none")
        else:
            tolerance = max(float(resolved["fps_tolerance"]), abs(float(expected_fps)) * float(resolved["fps_tolerance_ratio"]))
            if abs(float(actual_fps) - float(expected_fps)) > tolerance:
                fail(
                    "gate.fps_mismatch",
                    f"fps {actual_fps} differs from expected {expected_fps} beyond tolerance {tolerance:g}",
                    detail={"expected": expected_fps, "actual": actual_fps, "tolerance": tolerance},
                )

    features = summary.get("features") or []
    feature_keys = {str(item.get("key")) for item in features if isinstance(item, dict)}
    feature_kinds = {str(item.get("kind")) for item in features if isinstance(item, dict)}
    missing = [key for key in resolved["required_features"] if key not in feature_keys]
    if missing:
        fail("gate.required_features", f"missing required features: {missing}", detail={"missing": missing})
    missing_kinds = [kind for kind in resolved["required_feature_kinds"] if kind not in feature_kinds]
    if missing_kinds:
        fail(
            "gate.required_feature_kinds",
            f"missing required feature kinds: {missing_kinds}",
            detail={"missing": missing_kinds},
        )

    quality_thresholds = any(
        resolved[key] is not None
        for key in ("max_nan_ratio", "max_anomaly_windows", "max_error_anomalies", "max_jump_windows", "max_stall_windows")
    )
    quality = None
    if quality_thresholds or resolved["include_frames"]:
        report(62.0, "running quality checks")
        try:
            quality = quality_report(
                path,
                use_cache=bool(resolved["use_cache"]),
                chunk_frames=resolved["chunk_frames"],
                max_frames=resolved["max_frames"],
                progress=(lambda value, message: report(62.0 + value * 0.32, message)) if progress is not None else None,
            )
        except Exception as exc:
            fail(
                "gate.frames_unavailable",
                f"frame quality checks could not run: {type(exc).__name__}: {exc}",
                hint="remove frame thresholds for metadata-only formats or install the adapter dependency",
            )
        else:
            stats = quality.get("stats") or {}
            nan_ratios = [float(item.get("nan_ratio") or 0.0) for item in stats.values() if isinstance(item, dict)]
            worst_nan = max(nan_ratios, default=0.0)
            if resolved["max_nan_ratio"] is not None and worst_nan > float(resolved["max_nan_ratio"]):
                fail(
                    "gate.nan_ratio",
                    f"worst NaN ratio {worst_nan:.3%} exceeds {float(resolved['max_nan_ratio']):.3%}",
                    detail={"worst": worst_nan, "limit": resolved["max_nan_ratio"]},
                )
            windows = quality.get("anomalies") or []
            by_code: dict[str, int] = {}
            by_severity: dict[str, int] = {}
            for window in windows:
                by_code[str(window.get("code"))] = by_code.get(str(window.get("code")), 0) + 1
                by_severity[str(window.get("severity"))] = by_severity.get(str(window.get("severity")), 0) + 1
            checks = (
                ("max_anomaly_windows", len(windows), "gate.anomaly_windows", "anomaly windows"),
                ("max_error_anomalies", by_severity.get("error", 0), "gate.error_anomalies", "error anomalies"),
                ("max_jump_windows", by_code.get("jump", 0), "gate.jump_windows", "jump windows"),
                ("max_stall_windows", by_code.get("stall", 0), "gate.stall_windows", "stall windows"),
            )
            for rule, count, code, label in checks:
                limit = resolved[rule]
                if limit is not None and count > int(limit):
                    fail(code, f"{count} {label} exceed the limit {int(limit)}", detail={"count": count, "limit": limit})

    combined = [Issue(**issue) for issue in validation.get("issues", []) if isinstance(issue, dict)] + issues
    counts = summarize_issues(combined)
    report(100.0, "gate completed")
    return {
        "path": str(path),
        "format": validation.get("format"),
        "ok": not failures,
        "score": counts["score"],
        "rules": resolved,
        "gate": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
        "validation": validation,
        "quality": quality,
        "issues": [issue.to_dict() for issue in combined],
        **counts,
    }
