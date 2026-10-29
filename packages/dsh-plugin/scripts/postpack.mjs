#!/usr/bin/env node
/** Remove the staged Python tree after `npm pack` / `npm publish`. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PLUGIN_ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
fs.rmSync(path.join(PLUGIN_ROOT, 'python'), { recursive: true, force: true });
