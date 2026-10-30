"""Object-store URI parsing and atomic materialization."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from embodied_core.storage import StorageError, StorageUnavailable, parse_storage_uri, pull


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parse_local_and_windows_paths(self) -> None:
        parsed = parse_storage_uri(self.root / "dataset")
        self.assertEqual(parsed.scheme, "file")
        self.assertEqual(parse_storage_uri("s3://bucket/prefix").scheme, "s3")
        self.assertEqual(parse_storage_uri("https://example.com/a").scheme, "https")

    def test_local_directory_pull_is_checksummed_and_atomic(self) -> None:
        source = self.root / "source"
        source.mkdir()
        (source / "data.bin").write_bytes(b"abc")
        destination = self.root / "dest"
        result = pull(source, destination)
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["bytes"], 3)
        manifest = json.loads((destination / ".embodied-source.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["files"][0]["sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        with self.assertRaises(FileExistsError):
            pull(source, destination)
        with self.assertRaises(StorageError):
            pull(source, source / "nested-output")

    def test_http_single_file_pull(self) -> None:
        source = self.root / "serve"
        source.mkdir()
        (source / "episode.bin").write_bytes(b"payload")
        handler = partial(_QuietHandler, directory=str(source))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            uri = f"http://127.0.0.1:{server.server_address[1]}/episode.bin"
            destination = self.root / "downloaded.bin"
            result = pull(uri, destination)
            self.assertEqual(destination.read_bytes(), b"payload")
            self.assertEqual(result["scheme"], "http")
            self.assertTrue(Path(result["manifest"]).is_file())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_s3_without_boto3_is_structured(self) -> None:
        try:
            import boto3  # noqa: F401
        except ImportError:
            with self.assertRaises(StorageUnavailable) as caught:
                pull("s3://bucket/prefix", self.root / "s3out")
            self.assertIn("embodied-core[s3]", caught.exception.install or "")


if __name__ == "__main__":
    unittest.main()
