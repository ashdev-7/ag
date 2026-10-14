import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// base './' so the built site works from any path (GitHub Pages serves it
// under /<repo>/). the engine (public/engine/ag.mjs + ag.wasm) is copied as is
export default defineConfig({
  base: './',
  plugins: [react()],
  worker: { format: 'es' },
  build: { target: 'es2022' },
  test: { environment: 'node' },
});
