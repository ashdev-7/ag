import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// base './' so the built site works from any path (GitHub Pages serves it
// under /<repo>/). the engine (public/engine/ag.mjs + ag.wasm) is copied as is
export default defineConfig({
  base: './',
  plugins: [react()],
  worker: { format: 'es' },
  // old enough for Safari 14, Chrome 87, Firefox 78: the syntax is lowered,
  // and src/polyfills.ts fills in the newer built-ins the code uses
  build: { target: ['es2020', 'safari14', 'chrome87', 'firefox78', 'edge88'] },
  test: { environment: 'node' },
});
