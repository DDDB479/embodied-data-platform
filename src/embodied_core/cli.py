"""Command-line interface: ``embodied <command>``.

Commands are thin wrappers over the library so that scripts and agents get the
same behaviour as humans. Every command prints JSON with ``--json`` for tool
consumption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .annotations import apply_annotation_feedback
from .api import create_server
from .cache import QualityCache
from .catalog import (
    export_manifest,
    history as catalog_history,
    index as catalog_index,
    lineage as catalog_lineage,
    remove as catalog_remove,
    runs as catalog_runs,
    search as catalog_search,
    stats as catalog_stats,
    versions as catalog_versions,
)
from .convert import ConversionError, convert
from .gate import evaluate_quality_gate, load_quality_rules
from .media import probe_dataset_media
from .platform import capability_report
from .model import summarize_issues, to_jsonable
from .quality import build_labeling_tasks, compute_dataset_stats, detect_anomalies, make_windows, segment_by_motion
from .registry import UnsupportedFormat, describe_formats, detect_dataset, open_dataset
from .reports import quality_report
from .storage import parse_storage_uri, pull as storage_pull
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


def cmd_doctor(args: argparse.Namespace) -> int:
    payload = capability_report()
    _emit(payload, True)
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


def cmd_gate(args: argparse.Namespace) -> int:
    try:
        rules = load_quality_rules(args.rules)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    overrides: dict[str, Any] = {}
    if args.expected_fps is not None:
        overrides["expected_fps"] = args.expected_fps
    if args.fps_tolerance is not None:
        overrides["fps_tolerance"] = args.fps_tolerance
    if args.min_score is not None:
        overrides["min_score"] = args.min_score
    if args.max_nan_ratio is not None:
        overrides["max_nan_ratio"] = args.max_nan_ratio
    if args.require_feature:
        overrides["required_features"] = args.require_feature
    if args.frames:
        overrides["include_frames"] = True
    if args.no_cache:
        overrides["use_cache"] = False
    if args.chunk_frames is not None:
        overrides["chunk_frames"] = args.chunk_frames
    if args.max_frames is not None:
        overrides["max_frames"] = args.max_frames
    rules.update(overrides)
    try:
        payload = evaluate_quality_gate(args.path, rules=rules)
    except FileNotFoundError:
        print(f"error: no such path: {args.path}", file=sys.stderr)
        return 2
    except UnsupportedFormat as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if args.json:
        _emit(payload, True)
    else:
        print(f"{'PASS' if payload['ok'] else 'REJECT'}  {payload['path']}  score={payload['score']}")
        for failure in payload["gate"]["failures"]:
            print(f"  [{failure['code']}] {failure['message']}")
    return 0 if payload["ok"] else 1


def cmd_media(args: argparse.Namespace) -> int:
    dataset = _dataset_or_exit(args.path)
    try:
        payload = probe_dataset_media(dataset, max_videos=args.max_videos, probe=not args.no_probe)
    finally:
        dataset.close()
    _emit(payload, True)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    try:
        if args.labels:
            payload = apply_annotation_feedback(args.manifest, args.labels, out=args.out)
        else:
            payload = apply_annotation_feedback(args.manifest, [], out=args.out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(payload, True)
    return 0 if payload.get("ok") else 1


def cmd_storage(args: argparse.Namespace) -> int:
    try:
        if args.storage_command == "info":
            parsed = parse_storage_uri(args.uri)
            _emit({"uri": parsed.raw, "scheme": parsed.scheme, "location": parsed.location}, True)
            return 0
        result = storage_pull(args.uri, args.destination, overwrite=args.overwrite, retries=args.retries)
    except FileExistsError as exc:
        print(f"error: destination exists: {exc}; pass --overwrite", file=sys.stderr)
        return 5
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(result, True)
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    try:
        server = create_server(
            args.db,
            host=args.host,
            port=args.port,
            token=args.token,
            cors=args.cors,
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"serving {args.db} on http://{args.host}:{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    try:
        payload = quality_report(
            args.path,
            episodes=args.episodes or None,
            features=args.feature or None,
            z=args.z,
            include_segments=args.segments,
            skip_anomalies=args.no_anomalies,
            use_cache=not args.no_cache,
            cache_path=args.cache_path,
            refresh_cache=args.refresh_cache,
            chunk_frames=args.chunk_frames,
            max_frames=args.max_frames,
        )
    except FileNotFoundError:
        print(f"error: no such path: {args.path}", file=sys.stderr)
        return 2
    except UnsupportedFormat as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    _emit(payload, True)
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    cache = QualityCache(args.path)
    if args.cache_command == "clear":
        removed = cache.clear()
        _emit({"cleared": removed, **cache.stats()}, True)
    else:
        _emit(cache.stats(), True)
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
        result = convert(
            args.source,
            args.destination,
            target=args.target,
            overwrite=args.overwrite,
            format=args.format,
            episodes=[int(v) for v in (args.episodes or [])] or None,
            train_ratio=args.train_ratio,
            validation_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
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
    if args.catalog_command == "versions":
        rows = catalog_versions(args.db, path=args.path, limit=args.limit)
        _emit({"count": len(rows), "versions": rows}, True)
        return 0
    if args.catalog_command == "lineage":
        rows = catalog_lineage(args.db, path=args.path, version_id=args.version_id, limit=args.limit)
        _emit({"count": len(rows), "lineage": rows}, True)
        return 0
    if args.catalog_command == "runs":
        rows = catalog_runs(args.db, limit=args.limit, kind=args.kind, status=args.status)
        _emit({"count": len(rows), "runs": rows}, True)
        return 0
    if args.catalog_command == "history":
        _emit(catalog_history(args.db, path=args.path, limit=args.limit), True)
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

    p = sub.add_parser("doctor", help="show optional dependencies, decoder limits and platform capabilities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

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

    p = sub.add_parser("gate", help="evaluate configurable publish rules")
    p.add_argument("path")
    p.add_argument("--rules", help="JSON/TOML rule file")
    p.add_argument("--expected-fps", type=float)
    p.add_argument("--fps-tolerance", type=float)
    p.add_argument("--require-feature", action="append", help="required feature key (repeatable)")
    p.add_argument("--min-score", type=int)
    p.add_argument("--max-nan-ratio", type=float)
    p.add_argument("--frames", action="store_true", help="include a frame-level scan")
    p.add_argument("--chunk-frames", type=int, help="frames per streaming chunk")
    p.add_argument("--max-frames", type=int, help="maximum frames to materialize (0 = unlimited)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("media", help="inventory modalities and optionally probe video with ffprobe")
    p.add_argument("path")
    p.add_argument("--max-videos", type=int, default=4)
    p.add_argument("--no-probe", action="store_true", help="inventory only; never call ffprobe")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_media)

    p = sub.add_parser("quality", help="feature statistics and anomaly windows")
    p.add_argument("path")
    p.add_argument("--episodes", nargs="*", type=int, help="restrict to these episode indices")
    p.add_argument("--feature", action="append", help="restrict to these features (repeatable)")
    p.add_argument("--z", type=float, default=8.0, help="robust z threshold for jump detection")
    p.add_argument("--no-anomalies", action="store_true")
    p.add_argument("--segments", action="store_true", help="include motion segments")
    p.add_argument("--no-cache", action="store_true", help="recompute even when a cached report exists")
    p.add_argument("--cache-path", help="quality cache SQLite path (default: per-user cache dir)")
    p.add_argument("--refresh-cache", action="store_true", help="ignore and replace an existing cached report")
    p.add_argument("--chunk-frames", type=int, help="frames per streaming chunk (default: 50000)")
    p.add_argument("--max-frames", type=int, help="maximum frames to materialize (0 = unlimited)")
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

    p = sub.add_parser("review", help="reconcile annotation labels with a labeling manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels", help="labels JSON/JSONL; omit to report pending coverage")
    p.add_argument("--out", help="write the reviewed manifest JSON or labels JSONL")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("convert", help="convert between layouts")
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--target", default="jsonl", choices=["jsonl", "lerobot", "lerobot-v3"])
    p.add_argument("--format", help="force the source format")
    p.add_argument("--episodes", nargs="*", help="only these episodes (jsonl target)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--train-ratio", type=float, default=0.9, help="v3 train split ratio")
    p.add_argument("--val-ratio", type=float, default=0.0, help="v3 validation split ratio")
    p.add_argument("--test-ratio", type=float, default=0.0, help="v3 test split ratio")
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

    c = cat.add_parser("versions", parents=[common], help="list immutable dataset versions")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--path")
    c.add_argument("--limit", type=int, default=50)

    c = cat.add_parser("lineage", parents=[common], help="show version parent/child edges")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--path")
    c.add_argument("--version-id", type=int)
    c.add_argument("--limit", type=int, default=100)

    c = cat.add_parser("runs", parents=[common], help="show catalog run records")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--kind")
    c.add_argument("--status")
    c.add_argument("--limit", type=int, default=20)

    c = cat.add_parser("history", parents=[common], help="versions, lineage and runs together")
    c.add_argument("--db", default="catalog.db")
    c.add_argument("--path")
    c.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("storage", help="materialize local/HTTP/S3 datasets atomically")
    store = p.add_subparsers(dest="storage_command", required=True)
    c = store.add_parser("pull", help="download/copy a URI into a local dataset directory")
    c.add_argument("uri")
    c.add_argument("destination")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--retries", type=int, default=2)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_storage)
    c = store.add_parser("info", help="parse a storage URI")
    c.add_argument("uri")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_storage)

    p = sub.add_parser("api", help="serve the read-only HTTP Catalog API")
    p.add_argument("--db", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token", default=os.environ.get("EMBODIED_API_TOKEN"))
    p.add_argument("--cors", default=None)
    p.set_defaults(func=cmd_api)

    p = sub.add_parser("cache", help="inspect or clear the quality report cache")
    cache = p.add_subparsers(dest="cache_command", required=True)
    c = cache.add_parser("stats", help="show cache entries, hits and size")
    c.add_argument("--path", help="cache SQLite path (default: per-user cache dir)")
    c.set_defaults(func=cmd_cache)
    c = cache.add_parser("clear", help="delete cached reports")
    c.add_argument("--path", help="cache SQLite path (default: per-user cache dir)")
    c.set_defaults(func=cmd_cache)
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
