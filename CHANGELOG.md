# Changelog

## 0.1.0 - 2026-10-29

### Packaging

- The DSH npm package now bundles `python/embodied_core` during `prepack`, so a registry install no longer needs a repository checkout or a separate `pip install`.
- `resolveConfig` reports whether the Python root came from config, the bundled `python/` tree, a source checkout, or a `pythonPath`/`PYTHONPATH` install.
- Added `node tools/check_clean_install.mjs`, which packs the package, installs it into a temporary app, starts the MCP server and probes a fixture.

### Safety

- DSH tool registration now propagates MCP annotations: only `readOnlyHint: true` tools are concurrency-safe; write tools remain serial.
- Write destinations outside the DSH workspace require an `allowed-once` approval. Missing approval support, a denied request or read-only mode fails closed.
- Relative path arguments resolve against the session workspace rather than the Python child process's install directory.

### Engineering

- Added a GitHub Actions matrix for Windows/Linux, Python 3.10/3.12, optional-reader and minimal installs.
- Added Node tests for bundled-root discovery, path normalization, concurrency declarations and approval branches.
- Added packaging invariants to the Python test suite.
