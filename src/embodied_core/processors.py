"""Processor plugin protocol used by pipelines and custom platform code."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .media import probe_video
from .model import to_jsonable
from .registry import open_dataset

ProcessorFn = Callable[[dict[str, Any], dict[str, Any]], Mapping[str, Any]]


class ProcessorError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessorSpec:
    name: str
    version: str
    description: str
    fn: ProcessorFn

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "description": self.description}


_PROCESSORS: dict[str, ProcessorSpec] = {}


def register_processor(
    name: str,
    fn: ProcessorFn,
    *,
    version: str = "1",
    description: str = "",
    replace: bool = False,
) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("processor name is required")
    if key in _PROCESSORS and not replace:
        raise ValueError(f"processor {key!r} is already registered")
    _PROCESSORS[key] = ProcessorSpec(key, str(version), description, fn)


def load_processor(import_path: str, *, name: str | None = None, replace: bool = False) -> str:
    """Register ``module:function`` for local platform extensions."""

    module_name, separator, attribute = str(import_path).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("processor import path must look like 'module:function'")
    imported = importlib.import_module(module_name)
    fn = getattr(imported, attribute, None)
    if not callable(fn):
        raise ProcessorError(f"{import_path!r} does not resolve to a callable")
    key = name or attribute
    register_processor(key, fn, description=f"imported from {import_path}", replace=replace)
    return key


def get_processor(name: str) -> ProcessorSpec:
    spec = _PROCESSORS.get(str(name))
    if spec is None:
        raise ProcessorError(f"unknown processor {name!r}; available: {sorted(_PROCESSORS)}")
    return spec


def list_processors() -> list[dict[str, Any]]:
    return [spec.to_dict() for _, spec in sorted(_PROCESSORS.items())]


def run_processor(name: str, arguments: Mapping[str, Any], context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = get_processor(name)
    result = spec.fn(dict(arguments), dict(context or {}))
    if not isinstance(result, Mapping):
        raise ProcessorError(f"processor {name!r} must return a mapping, got {type(result).__name__}")
    return to_jsonable(dict(result))


def _dataset_inventory(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    path = arguments.get("path") or context.get("input")
    if not path:
        raise ValueError("dataset_inventory requires path or context.input")
    dataset = open_dataset(path)
    try:
        return {"path": str(path), "summary": dataset.summary.to_dict(include_episodes=False)}
    finally:
        dataset.close()


def _video_probe(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    path = arguments.get("path")
    if not path:
        raise ValueError("video_probe requires path")
    return probe_video(path)


def _video_frame_plan(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    path = arguments.get("path")
    if not path:
        raise ValueError("video_frame_plan requires path")
    metadata = probe_video(path)
    if not metadata.get("available"):
        return {"video": metadata, "frames": [], "count": 0}
    fps = arguments.get("fps")
    if fps is None:
        rate = str(metadata.get("avg_frame_rate") or "0/1")
        numerator, _, denominator = rate.partition("/")
        try:
            fps = float(numerator) / max(float(denominator or 1), 1.0)
        except ValueError:
            fps = 30.0
    duration = float(metadata.get("duration_s") or 0.0)
    max_frames = max(1, int(arguments.get("max_frames", 1000)))
    interval = max(float(arguments.get("interval_s", 0.0)), 0.0)
    timestamps = [round(index * interval, 6) for index in range(max_frames) if index * interval < duration] if interval else []
    return {"video": metadata, "fps": fps, "timestamps": timestamps, "count": len(timestamps) or int(metadata.get("nb_frames") or 0)}


def _noop(arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    return {"arguments": arguments, "context_keys": sorted(context)}


register_processor("dataset_inventory", _dataset_inventory, description="Summarize a dataset without frames")
register_processor("video_probe", _video_probe, description="Probe one media file with ffprobe")
register_processor("video_frame_plan", _video_frame_plan, description="Plan frame timestamps from ffprobe metadata")
register_processor("noop", _noop, description="Pass-through processor for testing and branching")
