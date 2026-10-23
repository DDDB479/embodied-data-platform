"""On-disk format adapters.

Each module exposes ``detect(root) -> Detection | None`` and
``load(root, detection) -> DatasetObject``. The registry in
``embodied_core.registry`` orders and dispatches them.
"""

from __future__ import annotations

__all__ = ["base", "hdf5", "jsonl", "lerobot", "mcap", "rlds", "rosbag"]
