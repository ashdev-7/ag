// Serves dist/ with the response headers vercel.json gives the live site, so
// the security policy can be tested before it ships: any resource it blocks
// shows up as a console error, which navigate.ts and the other checks fail on.
//
//   npm run build && node scripts/serve.ts [port]
//   node scripts/navigate.ts chromium http://localhost:4192/
//   CSP_NO_WASM=1 node scripts/serve.ts    (a policy that refuses WebAssembly)

import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';

const port = Number(process.argv[2] ?? 4192);
const root = join(import.meta.dirname, '..', 'dist');
const config = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'vercel.json'), 'utf8')) as {
  headers: { source: string; headers: { key: string; value: string }[] }[];
};
// vercel's "/(.*)" patterns are regular expressions over the path
const rules = config.headers.map((r) => ({ re: new RegExp(`^${r.source}$`), headers: r.headers }));

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.wasm': 'application/wasm', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.txt': 'text/plain',
};

createServer((req, res) => {
  const path = decodeURIComponent((req.url ?? '/').split('?')[0]);
  let file = normalize(join(root, path));
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  if (!existsSync(file) || statSync(file).isDirectory()) file = join(root, 'index.html');
  for (const r of rules) if (r.re.test(path)) for (const h of r.headers) {
    // CSP_NO_WASM=1: a stricter site policy that refuses WebAssembly, as some
    // company networks impose; the engine must fall back to its JavaScript build
    const value = process.env.CSP_NO_WASM && h.key === 'Content-Security-Policy' ? h.value.replace(" 'wasm-unsafe-eval'", '') : h.value;
    res.setHeader(h.key, value);
  }
  res.setHeader('Content-Type', TYPES[extname(file)] ?? 'application/octet-stream');
  createReadStream(file).pipe(res);
}).listen(port, () => console.log(`serving dist with vercel.json headers on http://localhost:${port}/`));
