"""CI quality gate using the configurable rule engine.

    python examples/quality_gate.py /data/datasets/night_run examples/quality_rules.json

Exit codes: 0 pass, 1 reject, 2 usage/rule error. The same rules can be used by
`embodied gate` or the `quality_gate` MCP tool.
"""
from __future__ import annotations

import json
import sys

from embodied_core import evaluate_quality_gate


def main(argv: list[str]) -> int:
    args = [value for value in argv[1:] if value != "--json"]
    as_json = "--json" in argv[1:]
    if not args:
        print("usage: quality_gate.py <dataset> [rules.json|rules.toml]", file=sys.stderr)
        return 2
    try:
        report = evaluate_quality_gate(args[0], rules=args[1] if len(args) > 1 else None)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if report["ok"]:
        print(f"PASS {args[0]} score={report['score']}")
        if as_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    print(f"REJECT {args[0]} score={report['score']}", file=sys.stderr)
    for failure in report["gate"]["failures"]:
        print(f"  [{failure['code']}] {failure['message']}", file=sys.stderr)
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
