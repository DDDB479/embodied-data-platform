"""Read-only HTTP Catalog API using the standard library.

The service is intentionally small and embeddable: bearer-token auth, CORS,
pagination and strong ETags are supported without FastAPI/Flask. Writes stay in
the CLI/MCP ingestion path so the HTTP surface cannot mutate the catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .catalog import get_dataset, lineage, runs, search, stats, versions


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def make_handler(db_path: str, *, token: str | None = None, cors: str | None = None, max_page: int = 200) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"embodied-catalog-api/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
            if os.environ.get("EMBODIED_API_LOG"):
                super().log_message(fmt, *args)

        def _authorized(self) -> bool:
            if not token:
                return True
            auth = self.headers.get("Authorization", "")
            api_key = self.headers.get("X-API-Key", "")
            return auth == f"Bearer {token}" or api_key == token

        def _send(self, status: int, payload: Any, *, etag: bool = True) -> None:
            body = _json_bytes(payload)
            digest = hashlib.sha256(body).hexdigest()
            if etag and self.headers.get("If-None-Match") == f'"{digest}"':
                self.send_response(304)
                self.send_header("ETag", f'"{digest}"')
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if etag:
                self.send_header("ETag", f'"{digest}"')
            if cors:
                self.send_header("Access-Control-Allow-Origin", cors)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": {"code": code, "message": message}}, etag=False)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            if cors:
                self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key, If-None-Match")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                self._error(401, "unauthorized", "provide Authorization: Bearer <token> or X-API-Key")
                return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path == "/health":
                    self._send(200, {"ok": True, "service": "embodied-catalog-api", "version": __version__})
                    return
                if path == "/v1/stats":
                    self._send(200, stats(db_path))
                    return
                if path == "/v1/datasets":
                    limit = _bounded_int(query.get("limit", ["50"])[0], 1, max_page, "limit")
                    offset = _bounded_int(query.get("offset", ["0"])[0], 0, 10_000_000, "offset")
                    rows = search(
                        db_path,
                        text=_first(query, "text"),
                        fmt=_first(query, "format"),
                        robot_type=_first(query, "robot_type"),
                        min_episodes=_optional_int(_first(query, "min_episodes"), "min_episodes"),
                        min_duration=_optional_float(_first(query, "min_duration"), "min_duration"),
                        min_score=_optional_int(_first(query, "min_score"), "min_score"),
                        limit=limit,
                        offset=offset,
                    )
                    self._send(200, {"count": len(rows), "limit": limit, "offset": offset, "results": rows})
                    return
                if path == "/v1/runs":
                    limit = _bounded_int(query.get("limit", ["20"])[0], 1, max_page, "limit")
                    self._send(200, {"runs": runs(db_path, limit=limit, kind=_first(query, "kind"), status=_first(query, "status"))})
                    return
                parts = [part for part in path.split("/") if part]
                if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "datasets":
                    dataset_id = _bounded_int(parts[2], 1, 2**63 - 1, "dataset_id")
                    row = get_dataset(db_path, dataset_id)
                    if row is None:
                        self._error(404, "not_found", f"dataset {dataset_id} is not in the catalog")
                        return
                    if len(parts) == 3:
                        self._send(200, row)
                    elif len(parts) == 4 and parts[3] == "versions":
                        self._send(200, {"dataset_id": dataset_id, "versions": versions(db_path, path=row["path"])})
                    elif len(parts) == 4 and parts[3] == "lineage":
                        self._send(200, {"dataset_id": dataset_id, "lineage": lineage(db_path, path=row["path"])})
                    else:
                        self._error(404, "not_found", "unknown catalog endpoint")
                    return
                self._error(404, "not_found", "unknown catalog endpoint")
            except FileNotFoundError:
                self._error(404, "catalog_not_found", f"catalog database not found: {db_path}")
            except ValueError as exc:
                self._error(400, "bad_request", str(exc))
            except Exception as exc:
                self._error(500, "internal_error", f"{type(exc).__name__}: {exc}")

    return Handler


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _bounded_int(value: str, minimum: int, maximum: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _optional_int(value: str | None, name: str) -> int | None:
    return None if value is None else _bounded_int(value, -2**63, 2**63 - 1, name)


def _optional_float(value: str | None, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def create_server(
    db_path: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    token: str | None = None,
    cors: str | None = None,
    max_page: int = 200,
) -> ThreadingHTTPServer:
    handler = make_handler(db_path, token=token, cors=cors, max_page=max_page)
    server = ThreadingHTTPServer((host, int(port)), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodied-api", description="Read-only HTTP Catalog API")
    parser.add_argument("--db", required=True, help="catalog SQLite path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=os.environ.get("EMBODIED_API_TOKEN"), help="bearer token (or EMBODIED_API_TOKEN)")
    parser.add_argument("--cors", help="Access-Control-Allow-Origin value")
    args = parser.parse_args(argv)
    server = create_server(args.db, host=args.host, port=args.port, token=args.token, cors=args.cors)
    print(f"serving catalog {args.db} on http://{args.host}:{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
