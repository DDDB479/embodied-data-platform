"""Read-only HTTP Catalog API: auth, pagination and ETags."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from embodied_core.api import create_server
from embodied_core.catalog import index

from tests import support


class CatalogAPITestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        datasets = self.root / "datasets"
        support.write_jsonl_dataset(datasets / "alpha")
        support.write_jsonl_dataset(datasets / "beta")
        self.db = self.root / "catalog.db"
        index(datasets, self.db)
        self.server = create_server(str(self.db), host="127.0.0.1", port=0, token="secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self._tmp.cleanup()

    def _get(self, path: str, *, token: str | None = "secret", headers: dict[str, str] | None = None) -> tuple[int, dict, dict]:
        request = urllib.request.Request(self.base + path)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, dict(response.headers), json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read()
            return exc.code, dict(exc.headers), json.loads(body) if body else {}

    def test_health_and_auth(self) -> None:
        status, _, payload = self._get("/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        status, _, payload = self._get("/health", token=None)
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["code"], "unauthorized")

    def test_pagination_and_etag(self) -> None:
        status, headers, payload = self._get("/v1/datasets?limit=1&offset=0")
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["limit"], 1)
        etag = headers["ETag"]
        status, _, payload = self._get("/v1/datasets?limit=1&offset=0", headers={"If-None-Match": etag})
        self.assertEqual(status, 304)
        self.assertEqual(payload, {})
        status, _, page2 = self._get("/v1/datasets?limit=1&offset=1")
        self.assertEqual(status, 200)
        self.assertNotEqual(payload, page2)

    def test_dataset_detail_versions_and_runs(self) -> None:
        _, _, page = self._get("/v1/datasets?limit=1")
        dataset_id = page["results"][0]["dataset_id"]
        status, _, detail = self._get(f"/v1/datasets/{dataset_id}")
        self.assertEqual(status, 200)
        self.assertEqual(detail["dataset_id"], dataset_id)
        status, _, versions = self._get(f"/v1/datasets/{dataset_id}/versions")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(versions["versions"]), 1)
        status, _, runs = self._get("/v1/runs")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(runs["runs"]), 1)

    def test_bad_request_is_structured(self) -> None:
        status, _, payload = self._get("/v1/datasets?limit=9999")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "bad_request")


if __name__ == "__main__":
    unittest.main()
