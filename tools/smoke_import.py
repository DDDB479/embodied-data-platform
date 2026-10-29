"""Smoke test: import every module and exercise the pure-python paths."""
from __future__ import annotations

import json
import sys
import traceback

sys.path.insert(0, "src")

failures = []


def step(label, fn):
    try:
        value = fn()
        print(f"[ok]   {label}")
        return value
    except Exception:
        print(f"[FAIL] {label}")
        traceback.print_exc()
        failures.append(label)
        return None


def t_import():
    import embodied_core
    from embodied_core import (  # noqa: F401
        DatasetSummary,
        EpisodeRecord,
        FeatureSpec,
        Issue,
        Severity,
        describe_formats,
        detect_dataset,
        open_dataset,
    )
    import embodied_core.catalog  # noqa: F401
    import embodied_core.cache  # noqa: F401
    import embodied_core.cli  # noqa: F401
    import embodied_core.convert  # noqa: F401
    import embodied_core.formats.hdf5  # noqa: F401
    import embodied_core.formats.jsonl  # noqa: F401
    import embodied_core.formats.lerobot  # noqa: F401
    import embodied_core.formats.mcap  # noqa: F401
    import embodied_core.formats.rlds  # noqa: F401
    import embodied_core.formats.rosbag  # noqa: F401
    import embodied_core.gate  # noqa: F401
    import embodied_core.mcp_server  # noqa: F401
    import embodied_core.jobs  # noqa: F401
    import embodied_core.quality  # noqa: F401
    import embodied_core.reports  # noqa: F401
    import embodied_core.registry  # noqa: F401
    import embodied_core.scaffold  # noqa: F401
    import embodied_core.validate  # noqa: F401
    return embodied_core.__version__


def t_formats():
    from embodied_core import describe_formats

    formats = describe_formats()
    assert len(formats) >= 6, formats
    for item in formats:
        assert {"format", "detect", "extensions"} <= set(item), item
    return [item["format"] for item in formats]


def t_score():
    from embodied_core import Issue, Severity, summarize_issues

    issues = [
        Issue(code="a", severity=Severity.ERROR, message="boom"),
        Issue(code="b", severity=Severity.WARN, message="meh"),
    ]
    result = summarize_issues(issues)
    assert result["errors"] == 1 and result["warnings"] == 1, result
    assert result["score"] == 100 - Severity.ERROR.weight - Severity.WARN.weight, result
    return result["score"]


def t_mcp_list():
    from embodied_core import mcp_server

    tools = mcp_server.list_tools()
    names = [tool["name"] for tool in tools]
    assert "validate_dataset" in names, names
    assert "quality_gate" in names and "catalog_history" in names, names
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object", tool
    response = mcp_server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert response["result"]["serverInfo"]["name"] == "embodied-data-studio"
    return len(names)


def t_cli_parser():
    from embodied_core.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["validate", "some/path", "--json"])
    assert args.command == "validate", args
    return sorted(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]


print("version:", step("import all modules", t_import))
print("formats:", step("describe_formats", t_formats))
print("score:", step("summarize_issues", t_score))
print("mcp tools:", step("mcp tools/list", t_mcp_list))
print("cli commands:", step("cli parser", t_cli_parser))

if failures:
    print(f"\n{len(failures)} failure(s): {failures}")
    raise SystemExit(1)
print("\nall smoke checks passed")
