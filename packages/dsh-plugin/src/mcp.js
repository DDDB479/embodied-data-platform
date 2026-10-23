/**
 * Minimal MCP stdio client.
 *
 * `embodied-mcp` speaks newline-delimited JSON-RPC 2.0 over stdin/stdout, so
 * the client only needs framing, request correlation, timeouts and a restart
 * path. No SDK dependency: this has to run inside a DSH plugin process that
 * must not pull arbitrary packages, and the same code is reused by the
 * `scripts/mcp-probe.mjs` diagnostics entry point.
 */
import { spawn } from 'node:child_process';
import path from 'node:path';

const PROTOCOL_VERSION = '2024-11-05';

/** Resolve an executable: absolute paths must exist, bare names go through PATH. */
function candidateCommand(name) {
  if (path.isAbsolute(name) || name.includes(path.sep) || name.includes('/')) {
    return name;
  }
  return name;
}

export class McpError extends Error {
  constructor(message, { code = 'MCP_ERROR', data = undefined } = {}) {
    super(message);
    this.name = 'McpError';
    this.code = code;
    this.data = data;
  }
}

export class McpClient {
  /**
   * @param {object} config resolved plugin config (see src/config.js)
   * @param {object} [logger] optional {info,warn,error} sink
   */
  constructor(config, logger = console) {
    this.config = config;
    this.logger = logger;
    this.child = null;
    this.buffer = '';
    this.pending = new Map();
    this.nextId = 1;
    this.ready = null;
    this.stderrTail = [];
    this.tools = null;
    this.lastError = null;
    this.disposed = false;
    this.spawnedWith = null;
  }

  /** Environment for the child process: inherit, then layer the config. */
  buildEnv() {
    const env = { ...process.env, ...this.config.env };
    const extra = this.config.pythonPath;
    if (extra.length > 0) {
      env.PYTHONPATH = [...extra, env.PYTHONPATH ?? ''].filter(Boolean).join(path.delimiter);
    }
    env.PYTHONUNBUFFERED = '1';
    env.PYTHONIOENCODING = 'utf-8';
    return env;
  }

  /** Try each configured interpreter until one exits cleanly for `--version`. */
  async pickPython() {
    const { pythonCandidates } = this.config;
    const failures = [];
    for (const candidate of pythonCandidates) {
      try {
        await runOnce(candidateCommand(candidate), ['--version'], { env: this.buildEnv(), cwd: this.config.cwd, timeoutMs: 15_000 });
        return candidate;
      } catch (error) {
        failures.push(`${candidate}: ${error.message}`);
      }
    }
    throw new McpError(
      `No usable Python interpreter. Tried ${pythonCandidates.length} candidate(s): ${failures.join(' | ')}. ` +
        'Set `python` in the plugin config (or EMBODIED_PYTHON) to an absolute interpreter path.',
      { code: 'NO_PYTHON' },
    );
  }

  /** Spawn the server once and complete the MCP handshake. */
  async start() {
    if (this.disposed) throw new McpError('Client disposed', { code: 'DISPOSED' });
    if (this.ready) return this.ready;
    this.ready = this.#bootstrap().catch((error) => {
      this.ready = null;
      this.lastError = error;
      throw error;
    });
    return this.ready;
  }

  async #bootstrap() {
    const python = await this.pickPython();
    this.spawnedWith = python;
    const child = spawn(python, this.config.args, {
      cwd: this.config.cwd,
      env: this.buildEnv(),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
    this.child = child;

    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk) => this.#onData(chunk));
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => this.#onStderr(chunk));
    child.once('error', (error) => this.#fail(new McpError(`Failed to start ${python}: ${error.message}`, { code: 'SPAWN_FAILED' })));
    child.once('exit', (code, signal) => {
      const detail = this.stderrTail.join('\n').trim();
      this.#fail(
        new McpError(
          `MCP server exited (code=${code ?? 'null'}, signal=${signal ?? 'none'})${detail ? `: ${detail}` : ''}`,
          { code: 'SERVER_EXIT' },
        ),
      );
    });

    await this.request('initialize', {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: 'dsh-embodied-data', version: '0.1.0' },
    }, { timeoutMs: this.config.startupTimeoutMs, skipEnsure: true });
    this.#write({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} });
    return child;
  }

  /** Cached `tools/list`; the tool surface is static for a given server build. */
  async listTools({ refresh = false } = {}) {
    if (this.tools && !refresh) return this.tools;
    // The MCP request timeout is a plugin-wide budget; discovery must not be
    // able to hang session bootstrap, hence the shorter cap.
    const result = await this.request('tools/list', {}, { timeoutMs: this.config.startupTimeoutMs });
    const tools = Array.isArray(result?.tools) ? result.tools : [];
    this.tools = tools;
    return tools;
  }

  /**
   * Invoke a tool and unwrap the MCP content envelope.
   * @returns {Promise<unknown>} parsed JSON payload, or `{ text }` for prose results
   */
  async callTool(name, args = {}, { timeoutMs, signal } = {}) {
    const result = await this.request(
      'tools/call',
      { name, arguments: args ?? {} },
      { timeoutMs: timeoutMs ?? this.config.timeoutMs, signal },
    );
    const payload = unwrapContent(result);
    if (result?.isError) {
      throw new McpError(typeof payload === 'string' ? payload : payload?.error ?? 'tool failed', { code: 'TOOL_ERROR', data: payload });
    }
    return payload;
  }

  /** Send a JSON-RPC request and await its matching response. */
  async request(method, params = {}, { timeoutMs = this.config.timeoutMs, signal, skipEnsure = false } = {}) {
    if (!skipEnsure) await this.start();
    if (this.disposed) throw new McpError('Client disposed', { code: 'DISPOSED' });
    const child = this.child;
    if (!child) throw new McpError('MCP server is not running', { code: 'NOT_RUNNING' });

    const id = this.nextId++;
    const message = { jsonrpc: '2.0', id, method, params };

    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new McpError(`${method} timed out after ${timeoutMs}ms`, { code: 'TIMEOUT' }));
      }, timeoutMs);
      if (typeof timer.unref === 'function') timer.unref();

      const onAbort = () => {
        cleanup();
        reject(new McpError(`${method} cancelled`, { code: 'ABORTED' }));
      };
      const cleanup = () => {
        clearTimeout(timer);
        this.pending.delete(id);
        if (signal) signal.removeEventListener('abort', onAbort);
      };
      if (signal) {
        if (signal.aborted) {
          cleanup();
          reject(new McpError(`${method} cancelled`, { code: 'ABORTED' }));
          return;
        }
        signal.addEventListener('abort', onAbort, { once: true });
      }

      this.pending.set(id, {
        resolve: (value) => {
          cleanup();
          resolve(value);
        },
        reject: (error) => {
          cleanup();
          reject(error);
        },
      });
      this.#write(message);
    });
  }

  #write(message) {
    const child = this.child;
    if (!child || !child.stdin.writable) {
      throw new McpError('MCP server stdin is closed', { code: 'NOT_RUNNING' });
    }
    child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  #onData(chunk) {
    this.buffer += chunk;
    let index = this.buffer.indexOf('\n');
    while (index >= 0) {
      const line = this.buffer.slice(0, index).trim();
      this.buffer = this.buffer.slice(index + 1);
      if (line) this.#onLine(line);
      index = this.buffer.indexOf('\n');
    }
    // A server that writes a partial line and dies should still surface it.
    if (this.buffer.length > 8 * 1024 * 1024) this.buffer = '';
  }

  #onLine(line) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      this.stderrTail.push(`unparsed stdout: ${line.slice(0, 400)}`);
      if (this.stderrTail.length > 20) this.stderrTail.shift();
      return;
    }
    if (Array.isArray(message)) {
      for (const item of message) this.#dispatch(item);
      return;
    }
    this.#dispatch(message);
  }

  #dispatch(message) {
    if (!message || typeof message !== 'object') return;
    if (message.id === undefined || message.id === null) return; // notification
    const entry = this.pending.get(message.id);
    if (!entry) return;
    if (message.error) {
      entry.reject(new McpError(message.error.message ?? 'JSON-RPC error', { code: String(message.error.code ?? 'RPC_ERROR'), data: message.error.data }));
      return;
    }
    entry.resolve(message.result);
  }

  #onStderr(chunk) {
    for (const line of chunk.split(/\r?\n/)) {
      if (!line.trim()) continue;
      this.stderrTail.push(line.trim());
      if (this.stderrTail.length > 40) this.stderrTail.shift();
    }
  }

  #fail(error) {
    this.lastError = error;
    this.ready = null;
    this.child = null;
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const entry of pending) entry.reject(error);
  }

  /** Terminate the child process; idempotent. */
  dispose() {
    this.disposed = true;
    const child = this.child;
    this.child = null;
    this.ready = null;
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const entry of pending) entry.reject(new McpError('Client disposed', { code: 'DISPOSED' }));
    if (child && child.exitCode === null && !child.killed) {
      try {
        child.stdin.end();
      } catch {
        /* already closed */
      }
      const timer = setTimeout(() => {
        try {
          child.kill();
        } catch {
          /* gone */
        }
      }, 1500);
      if (typeof timer.unref === 'function') timer.unref();
      child.once('exit', () => clearTimeout(timer));
    }
  }

  /** Snapshot for diagnostics / the self-check tool. */
  status() {
    return {
      running: Boolean(this.child),
      spawnedWith: this.spawnedWith,
      tools: this.tools ? this.tools.map((tool) => tool.name) : null,
      lastError: this.lastError ? this.lastError.message : null,
      stderrTail: this.stderrTail.slice(-5),
    };
  }
}

/** Turn the MCP content array into a plain JS value. */
export function unwrapContent(result) {
  const content = Array.isArray(result?.content) ? result.content : [];
  const texts = content.filter((item) => item && item.type === 'text' && typeof item.text === 'string').map((item) => item.text);
  if (texts.length === 0) return result?.structuredContent ?? null;
  if (texts.length === 1) {
    const only = texts[0];
    const trimmed = only.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        return JSON.parse(trimmed);
      } catch {
        return { text: only };
      }
    }
    return { text: only };
  }
  return { text: texts.join('\n') };
}

/** Run a process to completion, resolving on exit code 0. */
function runOnce(command, args, { env, cwd, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env, cwd, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    let output = '';
    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch {
        /* gone */
      }
      reject(new Error(`--version timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on('data', (chunk) => {
      output += String(chunk);
    });
    child.stderr.on('data', (chunk) => {
      output += String(chunk);
    });
    child.once('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once('exit', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(output.trim());
      else reject(new Error(`exit ${code}: ${output.trim().slice(0, 200)}`));
    });
  });
}
