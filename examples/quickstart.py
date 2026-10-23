"""Walk through the whole toolkit on a real (or generated) dataset.

    python examples/quickstart.py                    # generates a demo dataset
    python examples/quickstart.py /path/to/dataset   # uses your own

Nothing here is magic: every step is one public function from embodied_core.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from embodied_core import open_dataset, validate
from embodied_core.catalog import index as catalog_index
from embodied_core.catalog import search as catalog_search
from embodied_core.convert import convert
from embodied_core.quality import build_labeling_tasks, detect_anomalies, segment_by_motion
from embodied_core.registry import detect_dataset


def make_demo_dataset(root: Path) -> Path:
    """A tiny canonical JSONL dataset with one deliberate spike."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "format": "embodied-jsonl",
                "version": "1.0",
                "fps": 30.0,
                "robot_type": "so100",
                "tasks": ["pick up the cube"],
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [6]},
                    "action": {"dtype": "float32", "shape": [6]},
                },
            }
        ),
        encoding="utf-8",
    )
    with (root / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in range(3):
            rows = []
            for frame in range(60):
                phase = frame / 59
                spike = 40.0 if (episode == 1 and frame == 30) else 0.0
                rows.append(
                    {
                        "observation.state": [phase * 0.5 + spike, 0.1, 0.2, 0.0, 0.0, 0.0],
                        "action": [phase * 0.4 + spike, 0.1, 0.2, 0.0, 0.0, 0.0],
                        "timestamp": frame / 30.0,
                        "frame_index": frame,
                        "episode_index": episode,
                    }
                )
            handle.write(json.dumps({"episode_index": episode, "frames": rows}) + "\n")
    return root


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        path = Path(argv[1])
    else:
        path = make_demo_dataset(Path(tempfile.mkdtemp()) / "demo")
        print(f"generated demo dataset at {path}\n")

    # 1) What is this? Recognition never trusts the extension.
    detection = detect_dataset(path)
    print(f"[detect]   {detection.format} {detection.version} confidence={detection.confidence}")
    print(f"           evidence: {', '.join(detection.evidence)}")

    # 2) Schema first: who is in the dataset and what does each column mean.
    dataset = open_dataset(path)
    summary = dataset.summary
    print(f"[summarize] {summary.episode_count} episodes, {summary.total_frames} frames, fps={summary.fps}")
    for feature in summary.features:
        print(f"           {feature.key:<24} {feature.dtype}{feature.shape} kind={feature.kind}")

    # 3) Structural health. score = 100 - error*30 - warn*8 - info*1.
    report = validate(path)
    print(f"[validate] score={report['score']} ok={report['ok']} issues={len(report['issues'])}")
    for issue in report["issues"]:
        print(f"           [{issue['severity']}] {issue['code']}: {issue['message']}")

    # 4) Per-frame quality. Bookkeeping columns are skipped on purpose.
    ids, columns = dataset.frames()
    windows = detect_anomalies(ids, columns, fps=summary.fps)
    segments = segment_by_motion(ids, columns, fps=summary.fps)
    print(f"[quality]  {len(windows)} anomaly windows, {sum(len(v) for v in segments.values())} motion segments")
    for window in windows[:5]:
        span = f"{window['start_s']:.2f}s-{window['end_s']:.2f}s" if "start_s" in window else ""
        print(f"           ep{window['episode']} {window['code']} on {window['feature']} "
              f"frames {window['start_frame']}-{window['end_frame']} {span}")

    # 5) Turn findings into reviewable work.
    manifest = build_labeling_tasks(summary, windows, include_segments=True, segments=segments)
    print(f"[label]    {manifest['task_count']} tasks")
    for task in manifest["tasks"][:3]:
        print(f"           ep{task['episode']} {task['start_frame']}-{task['end_frame']}: {task['suggested_label']}")

    # 6) Convert for training. lerobot target needs pyarrow.
    out = path.parent / "converted_lerobot"
    try:
        result = convert(path, out, target="lerobot", overwrite=True)
        print(f"[convert]  {result['frames']} frames -> {out} ({result['target']})")
    except Exception as exc:  # missing optional dependency or bad target
        print(f"[convert]  skipped: {type(exc).__name__}: {exc}")

    # 7) Catalogue a tree, then query it.
    db = path.parent / "catalog.db"
    indexed = catalog_index(path.parent, db, force=True)
    hits = catalog_search(db, min_episodes=1, limit=5)
    print(f"[catalog]  indexed {indexed['added']} dataset(s), search returned {len(hits)}")
    for hit in hits[:3]:
        print(f"           {hit['format']:<16} {hit['episodes']:>4} episodes  {hit['path']}")

    dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
