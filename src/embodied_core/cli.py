"""Command-line interface: ``embodied <command>``.

Commands are thin wrappers over the library so that scripts and agents get the
same behaviour as humans. Every command prints JSON with ``--json`` for tool
consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .catalog import export_manifest, index as catalog_index, remove as catalog_remove, search as catalog_search, stats as catalog_stats
from .convert import ConversionError, convert
from .model import summarize_issues, to_jsonable
from .quality import build_labeling_tasks, compute_dataset_stats, detect_anomalies, make_windows, segment_by_motion
from .registry import UnsupportedFormat, describe_formats, detect_dataset, open_dataset
from .scaffold import scaffold
from .validate import CHECKS, DEFAULT_CHECKS, run_checks, validate


def _emit(payload: Any, as_json: bool, human: str | None = None) -> None:
    if as_json or human is None:
        print(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False))
    else:
        print(human)


def _dataset_or_exit(path: str):
    try:
        return open_dataset(path)
    except FileNotFoundError:
        print(f"error: no such path: {path}", file=sys.stderr)
        raise SystemExit(2)
    except UnsupportedFormat as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: run `embodied formats` to see what is supported", file=sys.stderr)
        raise SystemExit(3)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_formats(args: argparse.Namespace) -> int:
    rows = describe_formats()
    if args.json:
        _emit(rows, True)
        return 0
    print(f"{'format':<15} {'needs':<28} detect")
    for row in rows:
        print(f"{row['format']:<15} {row['needs'][:27]:<28} {row['detect']}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    try:
        detection = detect_dataset(args.path)
    except (FileNotFoundError, UnsupportedFormat) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(detection.to_dict(), args.json, f"{detection.format} (confidence {detection.confidence:.2f})")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    dataset = _dataset_or_exit(args.path)
    try:
        summary = dataset.summary
        payload = summary.to_dict(include_episodes=not args.no_episodes)
        payload["capabilities"] = sorted(dataset.capabilities)
        payload["optional_dependencies"] = dataset.optional_dependencies()
        if args.json:
            _emit(payload, True)
        else:
            lines = [
                f"path        {summary.path}",
                f"format      {summary.format} {summary.version or ''}".rstrip(),
                f"episodes    {summary.episode_count}",
                f"frames      {summary.total_frames if summary.total_frames is not None else '?'}",
                f"duration    {summary.total_duration_s:.1f}s" if summary.total_duration_s else "duration    ?",
                f"fps         {summary.fps or '?'}",
                f"robot       {summary.robot_type or '?'}",
                f"features    {len(summary.features)}: {', '.join(summary.feature_keys()[:8])}",
                f"tasks       {len(summary.tasks)}",
            ]
            if summary.notes:
                lines.append("notes       " + "; ".join(summary.notes[:3]))
            print("\n".join(lines))
    finally:
        dataset.close()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate(args.path, format=args.format, checks=args.check or None, include_frames=args.frames)
    if args.json:
        _emit(report, True)
    else:
        head = f"{report.get('format') or 'unknown'}  score={report.get('score')}  " \
               f"errors={report.get('errors')}  warnings={report.get('warnings')}  infos={report.get('infos')}"
        print(head)
        for issue in report["issues"]:
            location = f" episode={issue['episode']}" if "episode" in issue else ""
            feature = f" feature={issue['feature']}" if "feature" in issue else ""
            print(f"  [{issue['severity']:<5}] {issue['code']:<28}{location}{feature}  {issue['message']}")
            if issue.get("hint"):
                print(f"          hint: {issue['hint']}")
    return 0 if report.get("errors", 1) == 0 else 1


def cmd_quality(args: argparse.Namespace) -> int:
    dataset = _dataset_or_exit(args.path)
    try:
        try:
            ids, columns = dataset.frames(args.episodes or None)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
        payload: dict[str, Any] = {"stats": compute_dataset_stats(columns)}
        if not args.no_anomalies:
            payload["anomalies"] = detect_anomalies(ids, columns, fps=dataset.summary.fps, z=args.z, features=args.feature or None)
            payload["anomaly_counts"] = {
                key: sum(1 for item in payload["anomalies"] if item["severity"] == key) for key in ("error", "warn", "info")
            }
        if args.segments:
            payload["segments"] = segment_by_motion(ids, columns, feature=args.feature, fps=dataset.summary.fps)
        _emit(payload, args.json or True)
    finally:
        dataset.close()
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    dataset = _dataset_or_exit(args.path)
    try:
        ids, columns = dataset.frames([args.episode] if args.episode is not None else None)
        segments = segment_by_motion(ids, columns, feature=args.feature, fps=dataset.summary.fps, min_segment=args.min_frames)
        _emit({"episode": args.episode, "segments": segments}, True)
    finally:
        dataset.close()
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    dataset = _dataset_or_exit(args.path)
    try:
        try:
            ids, columns = dataset.frames()
        except Exception:
            ids, columns = [], {}
        windows = detect_anomalies(ids, columns, fps=dataset.summary.fps, z=args.z) if columns else []
        segments = segment_by_motion(ids, columns, fps=dataset.summary.fps) if (args.segments and columns) else None
        manifest = build_labeling_tasks(dataset.summary, windows, max_tasks=args.max_tasks, include_segments=args.segments, segments=segments)
        if args.out:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(to_jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {manifest['task_count']} tasks to {out}")
        else:
            _emit(manifest, True)
    finally:
        dataset.close()
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    try:
        result = convert(args.source, args.destination, target=args.target, overwrite=args.overwrite, format=args.format,
                         episodes=[int(v) for v in (args.episodes or [])] or None)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    _emit(result, True)  # the index report is always JSON
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    if args.catalog_command == "index":
        result = catalog_index(args.root, args.db, max_depth=args.depth, force=args.force, validate=args.validate)
        _emit(result, True)  # the index report is always JSON
        return 0
    if args.catalog_command == "search":
        rows = catalog_search(
            args.db,
            text=args.text,
            fmt=args.format,
            robot_type=args.robot,
            min_episodes=args.min_episodes,
            min_duration=args.min_duration,
            min_score=args.min_score,
            limit=args.limit,
            order_by=args.order_by,
            descending=not args.ascending,
        )
        if args.json:
            _emit({"count": len(rows), "results": rows}, True)
            return 0
        if not rows:
            print("no datasets matched")
            return 0
        print(f"{'episodes':>8} {'frames':>9} {'dur(s)':>9} {'score':>5}  {'format':<12} {'robot':<14} path")
        for row in rows:
            print(
                f"{row['episodes'] or 0:>8} {row['frames'] or 0:>9} {row['duration_s'] or 0:>9.1f} "
                f"{row['score'] if row['score'] is not None else '-':>5}  {row['format']:<12} {(row['robot_type'] or '-')[:13]:<14} {row['path']}"
            )
        return 0
    if args.catalog_command == "stats":
        _emit(catalog_stats(args.db), True)
        return 0
    if args.catalog_command == "remove":
        removed = catalog_remove(args.db, args.path)
        _emit({"removed": removed, "path": args.path}, True)
        return 0 if removed else 1
    if args.catalog_command == "export":
        manifest = export_manifest(args.db, limit=args.limit)
        if args.out:
            Path(args.out).expanduser().write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {manifest['count']} entries to {args.out}")
        else:
            _emit(manifest, True)
        return 0
    print("error: missing catalog subcommand", file=sys.stderr)
    return 2


def cmd_scaffold(args: argparse.Namespace) -> int:
    try:
        result = scaffold(args.destination, kind=args.kind, name=args.name, overwrite=args.overwrite)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    if args.json:
        _emit(result, True)
    else:
        print(f"created {result['kind']} skeleton at {result['destination']}")
        for path in result["files"]:
            print(f"  {path}")
        for step in result["next_steps"]:
            print(f"next: {step}")
    return 0


def cmd_windows(args: argparse.Namespace) -> int:
    dataset = _dataset_or_exit(args.path)
    try:
        lengths = [e.length or 0 for e in dataset.summary.episodes]
        plan = make_windows(lengths, args.window, args.stride)
        _emit({"window": args.window, "stride": args.stride or args.window, "count": len(plan), "windows": plan[: args.limit]}, True)
    finally:
        dataset.close()
    return 0


def cmd_checks(args: argparse.Namespace) -> int:
    _emit({"default": DEFAULT_CHECKS, "available": sorted(CHECKS)}, args.json or True)
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embodied", description="Embodied-AI dataset toolkit")
    parser.add_argument("--version", action="version", version=f"embodied-core {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("formats", help="list supported dataset layouts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_formats)

    p = sub.add_parser("detect", help="detect the format of a path")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("info", help="summarise a dataset")
    p.add_argument("path")
    p.add_argument("--no-episodes", action="store_true", help="omit the per-episode list")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("validate", help="run structural checks")
    p.add_argument("path")
    p.add_argument("--format", help="force a format instead of detecting")
    p.add_argument("--check", action="append", help="run only these checks (repeatable)")
    p.add_argument("--frames", action="store_true", help="also scan numeric frames (slower)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("quality", help="feature statistics and anomaly windows")
    p.add_argument("path")
    p.add_argument("--episodes", nargs="*", type=int, help="restrict to these episode indices")
    p.add_argument("--feature", action="append", help="restrict to these features (repeatable)")
    p.add_argument("--z", type=float, default=8.0, help="robust z threshold for jump detection")
    p.add_argument("--no-anomalies", action="store_true")
    p.add_argument("--segments", action="store_true", help="include motion segments")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("segment", help="split episodes into moving/idle segments")
    p.add_argument("--json", action="store_true")
    p.add_argument("path")
    p.add_argument("--episode", type=int)
    p.add_argument("--feature")
    p.add_argument("--min-frames", type=int, default=5)
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("label", help="generate an annotation task manifest")
    p.add_argument("--json", action="store_true")
    p.add_argument("path")
    p.add_argument("--out", help="write the manifest JSON here")
    p.add_argument("--max-tasks", type=int, default=100)
    p.add_argument("--z", type=float, default=8.0)
    p.add_argument("--segments", action="store_true", help="also ask for phase descriptions")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("convert", help="convert between layouts")
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--target", default="jsonl", choices=["jsonl", "lerobot"])
    p.add_argument("--format", help="force the source format")
    p.add_argument("--episodes", nargs="*", help="only these episodes (jsonl target)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("scaffold", help="write a project skeleton")
    p.add_argument("destination")
    p.add_argument("--kind", default="platform", choices=["platform", "adapter", "pipeline"])
    p.add_argument("--name", default="embodied-platform")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scaffold)

    p = sub.add_parser("windows", help="plan fixed-length training windows")
    p.add_argument("--json", action="store_true")
    p.add_argument("path")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--stride", type=int)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_windows)

    p = sub.add_parser("checks", help="list validation checks")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_checks)

    p = sub.add_parser("catalog", help="dataset catalog operations")
    cat = p.add_subparsers(dest="catalog_command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit JSON")

    c = cat.add_parser("index", parents=[common], help="scan a tree and update the catalog")
    c.add_argument("root")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--depth", type=int, default=4)
    c.add_argument("--force", action="store_true")
    c.add_argument("--validate", action="store_true", help="store a health score per dataset")

    c = cat.add_parser("search", parents=[common], help="query the catalog")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--text")
    c.add_argument("--format")
    c.add_argument("--robot")
    c.add_argument("--min-episodes", type=int)
    c.add_argument("--min-duration", type=float)
    c.add_argument("--min-score", type=int)
    c.add_argument("--limit", type=int, default=50)
    c.add_argument("--order-by", default="indexed_at")
    c.add_argument("--ascending", action="store_true")

    c = cat.add_parser("stats", parents=[common], help="catalog totals")
    c.add_argument("--db", default="catalog.db")

    c = cat.add_parser("remove", parents=[common], help="drop one dataset from the catalog")
    c.add_argument("path")
    c.add_argument("--db", default="catalog.db")

    c = cat.add_parser("export", parents=[common], help="export a manifest of catalog entries")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--limit", type=int, default=1000)
    c.add_argument("--out")
    p.set_defaults(func=cmd_catalog)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
