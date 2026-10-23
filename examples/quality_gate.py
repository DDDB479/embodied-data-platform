"""CI quality gate: fail the build when a dataset would poison training.

    python examples/quality_gate.py /data/datasets/night_run

Exit codes: 0 pass, 1 reject, 2 usage error. Wire it into a pipeline exactly
like a linter, or call the same three functions from your ingest service.
"""
from __future__ import annotations

import sys
from pathlib import Path

from embodied_core import open_dataset, validate
from embodied_core.quality import detect_anomalies, signal_features

MIN_SCORE = 80
MAX_FORBIDDEN_WINDOWS = 3


def gate(path: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    report = validate(path)
    if report["format"] is None:
        return False, [f"not a dataset: {path}"]
    if report["errors"]:
        reasons.append(f"{report['errors']} error-level issues")
    if report["score"] < MIN_SCORE:
        reasons.append(f"score {report['score']} < {MIN_SCORE}")

    dataset = open_dataset(path)
    summary = dataset.summary
    try:
        if not summary.features:
            reasons.append("no features declared")
        keys = {feature.key for feature in summary.features}
        for required in ("observation.state", "action"):
            if required not in keys:
                reasons.append(f"missing required feature {required}")

        if not reasons:
            try:
                ids, columns = dataset.frames()
            except Exception as exc:  # frames are optional for some formats
                print(f"note: frame scan skipped ({exc})")
            else:
                windows = detect_anomalies(ids, columns, fps=summary.fps)
                fatal = [w for w in windows if w["code"] in {"non_finite", "jump"}]
                if len(fatal) > MAX_FORBIDDEN_WINDOWS:
                    reasons.append(f"{len(fatal)} discontinuity windows (limit {MAX_FORBIDDEN_WINDOWS})")
                print(
                    f"scanned {len(signal_features(columns))} signal columns: "
                    f"{len(windows)} windows, {len(fatal)} discontinuities"
                )
    finally:
        dataset.close()

    return not reasons, reasons


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: quality_gate.py <dataset>", file=sys.stderr)
        return 2
    ok, reasons = gate(Path(argv[1]))
    if ok:
        print(f"PASS {argv[1]}")
        return 0
    print(f"REJECT {argv[1]}", file=sys.stderr)
    for reason in reasons:
        print(f"  - {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
