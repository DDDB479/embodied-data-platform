/**
 * Converts MCP tool descriptors into DSH tool descriptors.
 *
 * DSH tools are `{ name, description, parameters, output, execute }`; MCP
 * tools are `{ name, description, inputSchema }` plus a `tools/call` round
 * trip. The adapter keeps DSH-facing names prefixed (`embodied_*`) so they sit
 * next to `ops_*`/`pwsh_*` without collisions, while the MCP name is preserved
 * for forwarding.
 */
import { describeConfig } from './config.js';

/** Structural schema for every tool result: the raw payload rides in `data`. */
export const OUTPUT_SCHEMA = {
  type: 'object',
  properties: {
    tool: { type: 'string' },
    ok: { type: 'boolean' },
    data: {},
  },
  required: ['tool'],
  additionalProperties: true,
};

function clip(text, maxChars) {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n… [truncated ${text.length - maxChars} of ${text.length} characters; narrow the request to see the rest]`;
}

/** Render a payload as compact, scannable text for the transcript. */
export function formatPayload(data, maxChars = 24_000) {
  if (data === null || data === undefined) return '(no payload)';
  if (typeof data === 'string') return clip(data, maxChars);
  if (Array.isArray(data)) {
    if (data.length === 0) return '(empty list)';
    const rendered = data.every((item) => item && typeof item === 'object' && !Array.isArray(item))
      ? data.map((item, index) => `${index + 1}. ${JSON.stringify(item)}`).join('\n')
      : JSON.stringify(data, null, 2);
    return clip(rendered, maxChars);
  }
  return clip(JSON.stringify(data, null, 2), maxChars);
}

function sanitize(rawName, prefix) {
  const base = String(rawName).replace(/[^A-Za-z0-9_-]/g, '_');
  return base.startsWith(prefix) ? base : `${prefix}${base}`;
}

function ensureObjectSchema(schema) {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) {
    return { type: 'object', properties: {}, additionalProperties: false };
  }
  const copy = { ...schema };
  if (copy.type !== 'object') copy.type = 'object';
  if (!copy.properties) copy.properties = {};
  return copy;
}

function filterTools(tools, config) {
  const include = new Set(config.includeTools);
  const exclude = new Set(config.excludeTools);
  return tools.filter((tool) => {
    if (!tool || typeof tool.name !== 'string') return false;
    if (exclude.has(tool.name)) return false;
    if (include.size > 0 && !include.has(tool.name)) return false;
    return true;
  });
}

/**
 * @param {Array<{name:string,description?:string,inputSchema?:object}>} tools
 * @param {import('./mcp.js').McpClient} client
 * @param {object} config
 */
export function buildToolDefinitions(tools, client, config, logger = console) {
  return filterTools(tools, config).map((tool) => {
    const name = sanitize(tool.name, config.prefix);
    const description = tool.description || `Embodied data tool: ${tool.name}`;
    return {
      name,
      description,
      parameters: ensureObjectSchema(tool.inputSchema),
      output: {
        schema: OUTPUT_SCHEMA,
        render: (_args, value) => [{ type: 'text', text: formatPayload(value?.data ?? value, config.maxOutputChars) }],
      },
      isConcurrencySafe: () => true,
      async execute(args, exec = {}) {
        const input = args && typeof args === 'object' && !Array.isArray(args) ? args : {};
        try {
          const data = await client.callTool(tool.name, input, { signal: exec.signal });
          return { tool: tool.name, ok: true, data };
        } catch (error) {
          if (error && (error.code === 'ABORTED' || error.code === 'DISPOSED')) throw error;
          logger?.warn?.(`embodied-data: ${tool.name} failed: ${error?.message ?? error}`);
          throw new Error(`embodied-data: ${tool.name} failed: ${error?.message ?? error}`);
        }
      },
    };
  });
}

/**
 * A tool that never touches Python: it reports how the plugin resolved its
 * runtime, so a broken install is diagnosable from inside the agent session.
 */
export function selfCheckTool(config, client) {
  return {
    name: `${config.prefix}self_check`,
    description:
      'Report how the embodied-data plugin resolved its Python runtime, whether the MCP server is reachable, and which tools it exposes. Call this first when an embodied_* tool fails or is missing.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    output: {
      schema: OUTPUT_SCHEMA,
      render: (_args, value) => [{ type: 'text', text: formatPayload(value?.data ?? value, config.maxOutputChars) }],
    },
    isConcurrencySafe: () => false,
    async execute() {
      const status = client.status();
      const report = {
        tool: 'self_check',
        ok: status.running,
        data: describeConfig(config, {
          server: status,
          pythonInstalled: Boolean(config.pythonRoot),
          hint: status.running
            ? 'Runtime looks healthy.'
            : 'Python server is not running. Check `python` in the plugin config, then PYTHONPATH (config.pythonPath) and `python -m embodied_core.mcp_server --list-tools`.',
        }),
      };
      return report;
    },
  };
}
