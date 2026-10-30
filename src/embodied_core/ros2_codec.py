"""Dependency-free CDR decoding for common ROS 2 message types.

The module intentionally targets the message families used by embodied-AI
recordings: joint state, IMU, odometry, images/depth, point clouds, laser scans,
poses and scalar/string topics. Unknown custom types fail with a stable error
instead of returning silently wrong numbers.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class CDRError(ValueError):
    """Malformed or truncated CDR payload."""


class UnsupportedMessageType(ValueError):
    def __init__(self, type_name: str) -> None:
        self.type_name = type_name
        super().__init__(
            f"CDR decoding for {type_name!r} is not registered; "
            "use a supported common ROS 2 type or provide a project decoder"
        )


def max_message_bytes() -> int:
    try:
        return max(1, int(os.environ.get("EMBODIED_MAX_MESSAGE_BYTES", 64 * 1024 * 1024)))
    except ValueError:
        return 64 * 1024 * 1024


def max_point_count() -> int:
    try:
        return max(1, int(os.environ.get("EMBODIED_MAX_POINTS", 5_000_000)))
    except ValueError:
        return 5_000_000


class CDRReader:
    """Little/big-endian CDR reader with alignment relative to payload origin."""

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self.data = memoryview(data)
        if len(self.data) >= 4 and self.data[0] == 0 and self.data[1] in (0, 1):
            self.little = self.data[1] == 1
            self.pos = 4
            self.origin = 4
        else:
            self.little = True
            self.pos = 0
            self.origin = 0
        self.endian = "<" if self.little else ">"

    def _align(self, size: int) -> None:
        relative = self.pos - self.origin
        padding = (-relative) % size
        self.pos += padding
        if self.pos > len(self.data):
            raise CDRError("alignment advanced past the payload")

    def _take(self, size: int) -> memoryview:
        if size < 0 or self.pos + size > len(self.data):
            raise CDRError(f"truncated CDR payload: need {size} bytes at offset {self.pos}")
        view = self.data[self.pos : self.pos + size]
        self.pos += size
        return view

    def _unpack(self, fmt: str, size: int) -> Any:
        self._align(size)
        return struct.unpack(self.endian + fmt, self._take(size))[0]

    def u8(self) -> int:
        return int(self._take(1)[0])

    def i8(self) -> int:
        return struct.unpack("b", self._take(1))[0]

    def boolean(self) -> bool:
        return bool(self.u8())

    def u16(self) -> int:
        return int(self._unpack("H", 2))

    def i16(self) -> int:
        return int(self._unpack("h", 2))

    def u32(self) -> int:
        return int(self._unpack("I", 4))

    def i32(self) -> int:
        return int(self._unpack("i", 4))

    def u64(self) -> int:
        return int(self._unpack("Q", 8))

    def i64(self) -> int:
        return int(self._unpack("q", 8))

    def f32(self) -> float:
        return float(self._unpack("f", 4))

    def f64(self) -> float:
        return float(self._unpack("d", 8))

    def string(self) -> str:
        length = self.u32()
        if length == 0:
            return ""
        raw = bytes(self._take(length))
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def sequence(self, read_one: Callable[[], Any]) -> list[Any]:
        length = self.u32()
        if length > len(self.data):
            raise CDRError(f"sequence length {length} exceeds payload size")
        return [read_one() for _ in range(length)]

    def string_sequence(self) -> list[str]:
        return self.sequence(self.string)

    def float64_sequence(self) -> np.ndarray:
        return np.asarray(self.sequence(self.f64), dtype=np.float64)

    def float32_sequence(self) -> np.ndarray:
        return np.asarray(self.sequence(self.f32), dtype=np.float64)

    def uint8_sequence(self) -> bytes:
        length = self.u32()
        if length > max_message_bytes():
            raise CDRError(f"byte sequence {length} exceeds EMBODIED_MAX_MESSAGE_BYTES")
        return bytes(self._take(length))


@dataclass
class DecodedMessage:
    """Numeric columns plus text/meta decoded from one ROS 2 message."""

    type_name: str
    numeric: dict[str, Any] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        body = {
            "type": self.type_name,
            "numeric": _jsonable(self.numeric),
            "text": self.text,
            "meta": _jsonable(self.meta),
        }
        if include_arrays:
            body["arrays"] = {key: value.tolist() for key, value in self.arrays.items()}
        return body


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def normalize_type_name(type_name: str) -> str:
    text = str(type_name or "").strip()
    for prefix in ("ros2msg:", "ros2idl:", "ros1msg:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.replace("::", "/")
    parts = [part for part in text.split("/") if part and part != "msg"]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return text


def _header(reader: CDRReader) -> dict[str, Any]:
    sec = reader.i32()
    nanosec = reader.u32()
    frame_id = reader.string()
    return {"stamp_s": sec + nanosec / 1e9, "frame_id": frame_id}


def _vector3(reader: CDRReader) -> np.ndarray:
    return np.asarray([reader.f64(), reader.f64(), reader.f64()], dtype=np.float64)


def _quaternion(reader: CDRReader) -> np.ndarray:
    return np.asarray([reader.f64(), reader.f64(), reader.f64(), reader.f64()], dtype=np.float64)


def _pose(reader: CDRReader) -> np.ndarray:
    return np.concatenate([_vector3(reader), _quaternion(reader)])


def _twist(reader: CDRReader) -> np.ndarray:
    return np.concatenate([_vector3(reader), _vector3(reader)])


def _array_sequence(reader: CDRReader, read_value: Callable[[], Any], dtype: Any) -> np.ndarray:
    values = reader.sequence(read_value)
    return np.asarray(values, dtype=dtype)


def _decode_joint_state(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    names = reader.string_sequence()
    position = reader.float64_sequence()
    velocity = reader.float64_sequence()
    effort = reader.float64_sequence()
    numeric = {
        "position": position,
        "velocity": velocity,
        "effort": effort,
    }
    return DecodedMessage(
        "sensor_msgs/JointState",
        numeric=numeric,
        text={"name": ",".join(names)},
        meta={"header": header, "joint_count": len(names), "names": names},
        arrays={"position": position, "velocity": velocity, "effort": effort},
    )


def _decode_imu(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    orientation = _quaternion(reader)
    gyro = _vector3(reader)
    accel = _vector3(reader)
    return DecodedMessage(
        "sensor_msgs/Imu",
        numeric={"orientation": orientation, "angular_velocity": gyro, "linear_acceleration": accel},
        meta={"header": header},
        arrays={"orientation": orientation, "angular_velocity": gyro, "linear_acceleration": accel},
    )


def _decode_odometry(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    child = reader.string()
    pose = _pose(reader)
    twist = _twist(reader)
    return DecodedMessage(
        "nav_msgs/Odometry",
        numeric={"pose": pose, "twist": twist},
        text={"child_frame_id": child},
        meta={"header": header},
        arrays={"pose": pose, "twist": twist},
    )


def _decode_pose_stamped(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    pose = _pose(reader)
    return DecodedMessage(
        "geometry_msgs/PoseStamped",
        numeric={"pose": pose},
        meta={"header": header},
        arrays={"pose": pose},
    )


def _decode_pose(reader: CDRReader) -> DecodedMessage:
    pose = _pose(reader)
    return DecodedMessage("geometry_msgs/Pose", numeric={"pose": pose}, arrays={"pose": pose})


def _decode_twist(reader: CDRReader) -> DecodedMessage:
    twist = _twist(reader)
    return DecodedMessage("geometry_msgs/Twist", numeric={"twist": twist}, arrays={"twist": twist})


def _decode_vector3(reader: CDRReader) -> DecodedMessage:
    vector = _vector3(reader)
    return DecodedMessage("geometry_msgs/Vector3", numeric={"vector": vector}, arrays={"vector": vector})


def _decode_laser_scan(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    angle_min = reader.f32()
    angle_max = reader.f32()
    angle_increment = reader.f32()
    time_increment = reader.f32()
    scan_time = reader.f32()
    range_min = reader.f32()
    range_max = reader.f32()
    ranges = reader.float32_sequence()
    intensities = reader.float32_sequence()
    finite = ranges[np.isfinite(ranges)]
    numeric = {
        "angle_min": angle_min,
        "angle_max": angle_max,
        "range_min": range_min,
        "range_max": range_max,
        "range_mean": float(finite.mean()) if finite.size else math.nan,
        "range_min_observed": float(finite.min()) if finite.size else math.nan,
        "range_max_observed": float(finite.max()) if finite.size else math.nan,
        "range_count": int(ranges.size),
    }
    return DecodedMessage(
        "sensor_msgs/LaserScan",
        numeric=numeric,
        meta={
            "header": header,
            "angle_increment": angle_increment,
            "time_increment": time_increment,
            "scan_time": scan_time,
        },
        arrays={"ranges": ranges, "intensities": intensities},
    )


_DEPTH_ENCODINGS = {
    "16UC1": np.dtype("uint16"),
    "mono16": np.dtype("uint16"),
    "32FC1": np.dtype("float32"),
}


def _decode_image(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    height = reader.u32()
    width = reader.u32()
    encoding = reader.string()
    is_bigendian = reader.u8()
    step = reader.u32()
    data = reader.uint8_sequence()
    numeric: dict[str, Any] = {
        "height": height,
        "width": width,
        "step": step,
        "bytes": len(data),
    }
    meta: dict[str, Any] = {
        "header": header,
        "encoding": encoding,
        "is_bigendian": bool(is_bigendian),
        "kind": "depth" if encoding in _DEPTH_ENCODINGS else "image",
    }
    arrays: dict[str, np.ndarray] = {}
    depth_dtype = _DEPTH_ENCODINGS.get(encoding)
    if depth_dtype is not None and height and width:
        bpp = depth_dtype.itemsize
        if step >= width * bpp and len(data) >= step * height:
            dtype = depth_dtype.newbyteorder(">" if is_bigendian else "<")
            image = np.ndarray(
                shape=(height, width),
                dtype=dtype,
                buffer=data,
                strides=(step, bpp),
            ).astype(np.float64)
            finite = image[np.isfinite(image)]
            numeric.update(
                {
                    "depth_min": float(finite.min()) if finite.size else math.nan,
                    "depth_max": float(finite.max()) if finite.size else math.nan,
                    "depth_mean": float(finite.mean()) if finite.size else math.nan,
                    "depth_nonzero_ratio": float(np.mean(image != 0)),
                }
            )
            arrays["depth"] = image
    elif encoding in ("rgb8", "bgr8") and height and width:
        bpp = 3
        if step >= width * bpp and len(data) >= step * height:
            image = np.ndarray(shape=(height, width, bpp), dtype=np.uint8, buffer=data, strides=(step, bpp, 1))
            channels = image.mean(axis=(0, 1)) / 255.0
            order = ("r", "g", "b") if encoding == "rgb8" else ("b", "g", "r")
            numeric.update({f"channel_mean_{name}": float(value) for name, value in zip(order, channels)})
            arrays["image"] = image
    return DecodedMessage("sensor_msgs/Image", numeric=numeric, meta=meta, arrays=arrays)


def _decode_compressed_image(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    fmt = reader.string()
    data = reader.uint8_sequence()
    return DecodedMessage(
        "sensor_msgs/CompressedImage",
        numeric={"bytes": len(data)},
        text={"format": fmt},
        meta={"header": header, "kind": "image"},
    )


_POINT_DTYPES = {
    1: np.dtype("int8"),
    2: np.dtype("uint8"),
    3: np.dtype("int16"),
    4: np.dtype("uint16"),
    5: np.dtype("int32"),
    6: np.dtype("uint32"),
    7: np.dtype("float32"),
    8: np.dtype("float64"),
}


def _decode_pointcloud2(reader: CDRReader) -> DecodedMessage:
    header = _header(reader)
    height = reader.u32()
    width = reader.u32()
    field_count = reader.u32()
    if field_count > 128:
        raise CDRError(f"PointCloud2 field count {field_count} is unreasonable")
    fields: list[dict[str, Any]] = []
    for _ in range(field_count):
        fields.append(
            {
                "name": reader.string(),
                "offset": reader.u32(),
                "datatype": reader.u8(),
                "count": reader.u32(),
            }
        )
    is_bigendian = reader.u8()
    point_step = reader.u32()
    row_step = reader.u32()
    data = reader.uint8_sequence()
    is_dense = reader.u8()
    count = int(height) * int(width)
    if count > max_point_count():
        raise CDRError(f"PointCloud2 has {count} points, above EMBODIED_MAX_POINTS={max_point_count()}")
    numeric: dict[str, Any] = {
        "height": height,
        "width": width,
        "point_count": count,
        "point_step": point_step,
        "row_step": row_step,
        "bytes": len(data),
        "dense": bool(is_dense),
    }
    arrays: dict[str, np.ndarray] = {}
    coordinate_fields = {item["name"]: item for item in fields if item["name"] in ("x", "y", "z")}
    if count and len(coordinate_fields) == 3 and len(data) >= row_step * height:
        xyz = np.empty((count, 3), dtype=np.float64)
        valid = True
        for axis, name in enumerate(("x", "y", "z")):
            item = coordinate_fields[name]
            dtype = _POINT_DTYPES.get(int(item["datatype"]))
            if dtype is None or dtype.kind != "f" or int(item["count"]) != 1:
                valid = False
                break
            dtype = dtype.newbyteorder(">" if is_bigendian else "<")
            if int(item["offset"]) + dtype.itemsize > point_step:
                valid = False
                break
            values = np.ndarray(
                shape=(height, width),
                dtype=dtype,
                buffer=data,
                offset=int(item["offset"]),
                strides=(row_step, point_step),
            ).reshape(-1)
            xyz[:, axis] = values
        if valid:
            finite = np.isfinite(xyz).all(axis=1)
            points = xyz[finite]
            numeric["finite_points"] = int(points.shape[0])
            if points.size:
                numeric.update(
                    {
                        "x_min": float(points[:, 0].min()),
                        "x_max": float(points[:, 0].max()),
                        "y_min": float(points[:, 1].min()),
                        "y_max": float(points[:, 1].max()),
                        "z_min": float(points[:, 2].min()),
                        "z_max": float(points[:, 2].max()),
                    }
                )
            arrays["xyz"] = xyz
    return DecodedMessage(
        "sensor_msgs/PointCloud2",
        numeric=numeric,
        meta={"header": header, "fields": fields, "kind": "point_cloud"},
        arrays=arrays,
    )


def _decode_string(reader: CDRReader) -> DecodedMessage:
    value = reader.string()
    return DecodedMessage("std_msgs/String", text={"data": value}, meta={"kind": "language"})


def _decode_string_multi_array(reader: CDRReader) -> DecodedMessage:
    # std_msgs/StringMultiArray is not a core std_msgs type, but several robot
    # stacks publish it. Layout: layout, data[].
    dim_count = reader.u32()
    dims = []
    for _ in range(dim_count):
        dims.append({"label": reader.string(), "size": reader.u32(), "stride": reader.u32()})
    data_offset = reader.u32()
    values = reader.string_sequence()
    return DecodedMessage(
        "std_msgs/StringMultiArray",
        numeric={"data_offset": data_offset},
        text={"data": " | ".join(values)},
        meta={"dims": dims, "values": values, "kind": "language"},
    )


def _decode_numeric_array(reader: CDRReader, type_name: str, numeric_type: str) -> DecodedMessage:
    dim_count = reader.u32()
    dims = []
    for _ in range(dim_count):
        dims.append({"label": reader.string(), "size": reader.u32(), "stride": reader.u32()})
    data_offset = reader.u32()
    if numeric_type == "float32":
        values = reader.float32_sequence()
    elif numeric_type == "float64":
        values = reader.float64_sequence()
    else:
        values = np.asarray(reader.sequence(lambda: reader.i32()), dtype=np.float64)
    return DecodedMessage(
        type_name,
        numeric={"data": values, "data_offset": data_offset},
        meta={"dims": dims},
        arrays={"data": values},
    )


def _decode_scalar(reader: CDRReader, type_name: str, kind: str) -> DecodedMessage:
    if kind == "bool":
        value: Any = reader.boolean()
    elif kind == "byte":
        value = reader.u8()
    elif kind == "char":
        value = reader.u8()
    elif kind == "int8":
        value = reader.i8()
    elif kind == "uint8":
        value = reader.u8()
    elif kind == "int16":
        value = reader.i16()
    elif kind == "uint16":
        value = reader.u16()
    elif kind == "int32":
        value = reader.i32()
    elif kind == "uint32":
        value = reader.u32()
    elif kind == "int64":
        value = reader.i64()
    elif kind == "uint64":
        value = reader.u64()
    elif kind == "float32":
        value = reader.f32()
    elif kind == "float64":
        value = reader.f64()
    else:
        raise UnsupportedMessageType(type_name)
    return DecodedMessage(type_name, numeric={"data": value}, arrays={"data": np.asarray([value], dtype=np.float64)})


_DECODERS: dict[str, Callable[[CDRReader], DecodedMessage]] = {
    "sensor_msgs/JointState": _decode_joint_state,
    "sensor_msgs/Imu": _decode_imu,
    "sensor_msgs/Image": _decode_image,
    "sensor_msgs/CompressedImage": _decode_compressed_image,
    "sensor_msgs/PointCloud2": _decode_pointcloud2,
    "sensor_msgs/LaserScan": _decode_laser_scan,
    "nav_msgs/Odometry": _decode_odometry,
    "geometry_msgs/PoseStamped": _decode_pose_stamped,
    "geometry_msgs/Pose": _decode_pose,
    "geometry_msgs/Twist": _decode_twist,
    "geometry_msgs/Vector3": _decode_vector3,
    "std_msgs/String": _decode_string,
    "std_msgs/StringMultiArray": _decode_string_multi_array,
    "std_msgs/Float32MultiArray": lambda reader: _decode_numeric_array(reader, "std_msgs/Float32MultiArray", "float32"),
    "std_msgs/Float64MultiArray": lambda reader: _decode_numeric_array(reader, "std_msgs/Float64MultiArray", "float64"),
    "std_msgs/Int32MultiArray": lambda reader: _decode_numeric_array(reader, "std_msgs/Int32MultiArray", "int32"),
}

for _name in ("Bool", "Byte", "Char", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64", "Float32", "Float64"):
    _kind = "bool" if _name == "Bool" else _name[0].lower() + _name[1:]
    _DECODERS[f"std_msgs/{_name}"] = (lambda reader, kind=_kind, name=_name: _decode_scalar(reader, f"std_msgs/{name}", kind))


def supported_message_types() -> list[str]:
    return sorted(_DECODERS)


def decode_message(type_name: str, data: bytes | bytearray | memoryview, *, encoding: str = "cdr") -> DecodedMessage:
    """Decode one CDR payload using its ROS 2 message type."""

    normalized_encoding = str(encoding or "cdr").lower()
    if normalized_encoding not in ("cdr", "ros2", "ros2msg"):
        raise FramesUnavailableEncoding(normalized_encoding)
    if len(data) > max_message_bytes():
        raise CDRError(f"message is {len(data)} bytes, above EMBODIED_MAX_MESSAGE_BYTES={max_message_bytes()}")
    normalized = normalize_type_name(type_name)
    decoder = _DECODERS.get(normalized)
    if decoder is None:
        raise UnsupportedMessageType(normalized)
    return decoder(CDRReader(data))


class FramesUnavailableEncoding(ValueError):
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        super().__init__(f"message encoding {encoding!r} is not CDR; only CDR/ROS2 payloads are decoded")


def flatten_decoded(decoded: DecodedMessage, prefix: str) -> dict[str, np.ndarray]:
    """Flatten a decoded message into frame columns keyed by ``prefix.subkey``."""

    out: dict[str, np.ndarray] = {}
    for key, value in decoded.numeric.items():
        if isinstance(value, np.ndarray):
            out[f"{prefix}.{key}"] = np.asarray(value, dtype=np.float64)
        elif isinstance(value, (int, float, bool)) and value is not None:
            out[f"{prefix}.{key}"] = np.asarray([float(value)], dtype=np.float64)
    return out


def topic_feature_keys(type_name: str) -> list[str]:
    """Names of numeric leaf columns a type can contribute (best effort)."""

    normalized = normalize_type_name(type_name)
    if normalized == "sensor_msgs/JointState":
        return ["position", "velocity", "effort"]
    if normalized == "sensor_msgs/Imu":
        return ["orientation", "angular_velocity", "linear_acceleration"]
    if normalized == "nav_msgs/Odometry":
        return ["pose", "twist"]
    if normalized in ("geometry_msgs/PoseStamped", "geometry_msgs/Pose"):
        return ["pose"]
    if normalized == "geometry_msgs/Twist":
        return ["twist"]
    if normalized == "geometry_msgs/Vector3":
        return ["vector"]
    if normalized == "sensor_msgs/Image":
        return ["height", "width", "step", "bytes", "depth_min", "depth_max", "depth_mean", "depth_nonzero_ratio"]
    if normalized == "sensor_msgs/PointCloud2":
        return ["point_count", "finite_points", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    if normalized == "sensor_msgs/LaserScan":
        return ["range_mean", "range_min_observed", "range_max_observed", "range_count"]
    if normalized in ("std_msgs/Float32MultiArray", "std_msgs/Float64MultiArray", "std_msgs/Int32MultiArray"):
        return ["data"]
    if normalized == "std_msgs/String":
        return []
    if normalized.startswith("std_msgs/"):
        return ["data"]
    return []
