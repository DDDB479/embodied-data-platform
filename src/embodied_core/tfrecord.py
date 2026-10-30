"""Minimal TFRecord + TensorFlow Example/SequenceExample decoding.

This is deliberately a *reader*: it covers the protobuf subset used by RLDS
without importing TensorFlow or protobuf. Resource limits protect the platform
from malformed length prefixes and oversized records.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class TFRecordError(ValueError):
    pass


def max_record_bytes() -> int:
    try:
        return max(1, int(os.environ.get("EMBODIED_MAX_RECORD_BYTES", 256 * 1024 * 1024)))
    except ValueError:
        return 256 * 1024 * 1024


@dataclass
class FeatureValue:
    kind: str
    values: list[Any]


@dataclass
class RLDSEpisode:
    fields: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)
    length: int | None = None
    encoding: str | None = None

    def numeric_columns(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in self.fields.items():
            array = _as_numeric(value)
            if array is not None:
                out[key] = array
        return out

    def texts(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in self.fields.items():
            if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                out[key] = " | ".join(value)
            elif isinstance(value, str):
                out[key] = value
        return out


def _read_varint(data: bytes | memoryview, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(data):
        byte = int(data[offset])
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7
        if shift > 70:
            raise TFRecordError("varint is too long")
    raise TFRecordError("truncated varint")


def _iter_fields(data: bytes | memoryview) -> Iterator[tuple[int, int, Any]]:
    offset = 0
    size = len(data)
    while offset < size:
        key, offset = _read_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            if offset + 8 > size:
                raise TFRecordError("truncated fixed64 field")
            value = bytes(data[offset : offset + 8])
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            if length < 0 or offset + length > size:
                raise TFRecordError("truncated length-delimited field")
            value = bytes(data[offset : offset + length])
            offset += length
        elif wire_type == 5:
            if offset + 4 > size:
                raise TFRecordError("truncated fixed32 field")
            value = bytes(data[offset : offset + 4])
            offset += 4
        else:
            raise TFRecordError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _decode_bytes_list(data: bytes) -> FeatureValue:
    values: list[bytes] = []
    for field, wire, value in _iter_fields(data):
        if field == 1 and wire == 2:
            values.append(value)
    return FeatureValue("bytes", values)


def _decode_float_list(data: bytes) -> FeatureValue:
    values: list[float] = []
    for field, wire, value in _iter_fields(data):
        if field != 1:
            continue
        if wire == 2:
            if len(value) % 4:
                raise TFRecordError("packed float list has a partial value")
            values.extend(struct.unpack(f"<{len(value) // 4}f", value))
        elif wire == 5:
            values.append(struct.unpack("<f", value)[0])
    return FeatureValue("float", values)


def _decode_int64_list(data: bytes) -> FeatureValue:
    values: list[int] = []
    for field, wire, value in _iter_fields(data):
        if field != 1:
            continue
        if wire == 0:
            values.append(int(value))
        elif wire == 2:
            offset = 0
            while offset < len(value):
                item, offset = _read_varint(value, offset)
                values.append(item)
    return FeatureValue("int", values)


def _decode_feature(data: bytes) -> FeatureValue:
    for field, wire, value in _iter_fields(data):
        if wire != 2:
            continue
        if field == 1:
            return _decode_bytes_list(value)
        if field == 2:
            return _decode_float_list(value)
        if field == 3:
            return _decode_int64_list(value)
    return FeatureValue("unknown", [])


def _decode_features(data: bytes) -> dict[str, FeatureValue]:
    out: dict[str, FeatureValue] = {}
    for field, wire, entry in _iter_fields(data):
        if field != 1 or wire != 2:
            continue
        key = ""
        feature = FeatureValue("unknown", [])
        for child_field, child_wire, child in _iter_fields(entry):
            if child_field == 1 and child_wire == 2:
                key = child.decode("utf-8", errors="replace")
            elif child_field == 2 and child_wire == 2:
                feature = _decode_feature(child)
        if key:
            out[key] = feature
    return out


def _decode_feature_list(data: bytes) -> list[FeatureValue]:
    values: list[FeatureValue] = []
    for field, wire, value in _iter_fields(data):
        if field == 1 and wire == 2:
            values.append(_decode_feature(value))
    return values


def _decode_feature_lists(data: bytes) -> dict[str, list[FeatureValue]]:
    out: dict[str, list[FeatureValue]] = {}
    for field, wire, entry in _iter_fields(data):
        if field != 1 or wire != 2:
            continue
        key = ""
        values: list[FeatureValue] = []
        for child_field, child_wire, child in _iter_fields(entry):
            if child_field == 1 and child_wire == 2:
                key = child.decode("utf-8", errors="replace")
            elif child_field == 2 and child_wire == 2:
                values = _decode_feature_list(child)
        if key:
            out[key] = values
    return out


def parse_example(data: bytes) -> dict[str, FeatureValue]:
    """Parse ``tensorflow.Example`` into a feature map."""

    features: dict[str, FeatureValue] = {}
    for field, wire, value in _iter_fields(data):
        if field == 1 and wire == 2:
            features = _decode_features(value)
    return features


def parse_sequence_example(data: bytes) -> tuple[dict[str, FeatureValue], dict[str, list[FeatureValue]]]:
    """Parse ``tensorflow.SequenceExample`` into context and feature lists."""

    context: dict[str, FeatureValue] = {}
    feature_lists: dict[str, list[FeatureValue]] = {}
    for field, wire, value in _iter_fields(data):
        if field == 1 and wire == 2:
            context = _decode_features(value)
        elif field == 2 and wire == 2:
            feature_lists = _decode_feature_lists(value)
    return context, feature_lists


def _feature_to_sequence(feature: FeatureValue) -> Any:
    if feature.kind == "bytes":
        decoded: list[Any] = []
        for value in feature.values:
            try:
                decoded.append(value.decode("utf-8"))
            except UnicodeDecodeError:
                decoded.append(value)
        return decoded
    return feature.values


def _features_to_fields(features: dict[str, FeatureValue]) -> dict[str, Any]:
    return {key: _feature_to_sequence(value) for key, value in features.items()}


def _feature_list_to_array(features: list[FeatureValue]) -> Any:
    if not features:
        return []
    kinds = {feature.kind for feature in features}
    if len(kinds) != 1:
        return [_feature_to_sequence(feature) for feature in features]
    kind = next(iter(kinds))
    if kind in ("float", "int"):
        rows = [[float(value) for value in feature.values] for feature in features]
        if all(len(row) == len(rows[0]) for row in rows):
            return np.asarray(rows, dtype=np.float64)
        return rows
    values = [_feature_to_sequence(feature) for feature in features]
    if all(isinstance(value, list) and len(value) == 1 for value in values):
        return [value[0] for value in values]
    return values


def _as_numeric(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
        array = value
    elif isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception:
            return None
    else:
        return None
    if array.dtype.kind not in "fiub" or array.ndim == 0:
        return None
    if array.ndim == 1:
        return array.reshape(-1, 1).astype(np.float64, copy=False)
    if array.ndim == 2:
        return array.astype(np.float64, copy=False)
    return array.reshape(array.shape[0], -1).astype(np.float64, copy=False)


def decode_rlds_record(data: bytes) -> RLDSEpisode:
    """Decode one RLDS TFRecord into step-aligned fields."""

    context, feature_lists = parse_sequence_example(data)
    if feature_lists:
        fields = {key: _feature_list_to_array(values) for key, values in feature_lists.items()}
        lengths = [len(value) for value in fields.values() if isinstance(value, (list, np.ndarray))]
        return RLDSEpisode(
            fields=fields,
            context=_features_to_fields(context),
            length=max(lengths) if lengths else None,
            encoding="sequence_example",
        )

    example = parse_example(data)
    if example:
        steps = example.get("steps")
        if steps is not None and steps.kind == "bytes" and len(steps.values) == 1:
            nested_context, nested_lists = parse_sequence_example(steps.values[0])
            if nested_lists:
                fields = {key: _feature_list_to_array(values) for key, values in nested_lists.items()}
                lengths = [len(value) for value in fields.values() if isinstance(value, (list, np.ndarray))]
                return RLDSEpisode(
                    fields=fields,
                    context=_features_to_fields(nested_context),
                    length=max(lengths) if lengths else None,
                    encoding="sequence_example",
                )
        fields = _features_to_fields(example)
        lengths = [len(value) for value in fields.values() if isinstance(value, (list, np.ndarray))]
        return RLDSEpisode(fields=fields, length=max(lengths) if lengths else None, encoding="example")
    raise TFRecordError("record is not a TF Example or SequenceExample with features")


def iter_tfrecords(path: str | Path) -> Iterator[bytes]:
    """Yield records from a TFRecord file, validating framing lengths."""

    target = Path(path)
    size = target.stat().st_size
    with target.open("rb") as handle:
        while handle.tell() + 12 <= size:
            header = handle.read(12)
            if len(header) < 12:
                break
            length = struct.unpack("<Q", header[:8])[0]
            if length <= 0 or length > max_record_bytes() or handle.tell() + length + 4 > size:
                raise TFRecordError(f"{target.name}: invalid TFRecord length {length}")
            payload = handle.read(length)
            handle.seek(4, 1)
            yield payload
