// Makes the link-preview image (public/og.png, 1200 x 630) that chat apps and
// social sites show when the site's link is shared. The picture on the right
// is a real decision boundary: the Spirals lab, trained by the engine. Also
// makes public/apple-touch-icon.png from public/favicon.svg.
//
//   npm run build && npm run og

import { spawn, spawnSync } from 'node:child_process';
import { join } from 'node:path';
import { chromium } from 'playwright';

const PORT = 4174;
const win = process.platform === 'win32';
const base = `http://localhost:${PORT}/`;
const out = join(import.meta.dirname, '../public/og.png');
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });
try {
  for (let i = 0; i < 40; ++i) {
    try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ }
    await sleep(250);
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

  // train the spirals until the boundary is clean
  await page.goto(`${base}#preset=spirals`);
  await page.waitForSelector('.boundary canvas');
  await page.getByLabel('Training speed').fill('10');   // the fastest
  await page.getByRole('button', { name: 'Run' }).click();
  await sleep(12000);
  await page.getByRole('button', { name: 'Pause' }).click();
  await sleep(500);
  const boundary = (await page.locator('.boundary canvas').first().screenshot()).toString('base64');
  const step = (await page.locator('.toolbar .step-count').textContent()) ?? '';
  console.log(`boundary captured at ${step}`);

  await page.setViewportSize({ width: 1200, height: 630 });
  await page.setContent(card(boundary));
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: 1200, height: 630 }, scale: 'css' });
  console.log(`wrote ${out}`);

  // the home-screen icon for iPhones, from the favicon
  const icon = join(import.meta.dirname, '../public/apple-touch-icon.png');
  await page.setViewportSize({ width: 180, height: 180 });
  await page.setContent(`<body style="margin:0"><img src="${base}favicon.svg" width="180" height="180" style="display:block"></body>`);
  await page.waitForFunction(() => document.images[0].complete);
  await page.screenshot({ path: icon, omitBackground: true, scale: 'css' });
  console.log(`wrote ${icon}`);
  await browser.close();
} finally {
  if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}

function card(png: string) {
  return `<!doctype html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; }
  body { width: 1200px; height: 630px; font-family: Inter, sans-serif; color: #e8eef9; background-color: #070b14;
         background-image: radial-gradient(700px 420px at 78% 10%, rgba(139,123,255,0.28), transparent 65%),
           radial-gradient(600px 400px at 0% 100%, rgba(77,163,255,0.14), transparent 60%),
           linear-gradient(rgba(148,170,220,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(148,170,220,0.05) 1px, transparent 1px);
         background-size: auto, auto, 44px 44px, 44px 44px;
         display: grid; grid-template-columns: 1fr 470px; gap: 56px; align-items: center; padding: 0 72px;
         -webkit-font-smoothing: antialiased; position: relative; overflow: hidden; }
  .eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 16px; letter-spacing: 0.22em; text-transform: uppercase; color: #5be9ff; }
  .brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 30px; letter-spacing: -0.03em; margin-top: 16px; }
  .brand span { color: #b3a8ff; font-weight: 500; }
  h1 { font-family: 'Space Grotesk', sans-serif; font-size: 70px; line-height: 1.0; letter-spacing: -0.045em; font-weight: 700; margin: 26px 0 22px; }
  h1 span { background: linear-gradient(92deg, #5be9ff, #b3a8ff 55%, #f5bcd9); -webkit-background-clip: text; background-clip: text; color: transparent; }
  p { font-size: 24px; line-height: 1.45; color: #a3b0c7; max-width: 540px; }
  .tags { display: flex; gap: 10px; margin-top: 30px; }
  .tags b { font-weight: 500; font-size: 17px; color: #d6d0ff; background: rgba(139,123,255,0.14); border: 1px solid rgba(139,123,255,0.35); padding: 7px 14px; border-radius: 999px; }
  .pic { background: rgba(16,22,38,0.8); border: 1px solid rgba(148,170,220,0.18); border-radius: 22px; padding: 18px;
         box-shadow: 0 30px 80px rgba(0,0,0,0.6), 0 0 60px rgba(139,123,255,0.22); }
  .crop { overflow: hidden; border-radius: 12px; }
  .crop img { display: block; width: 100%; transform: scale(1.28); }   /* the spirals fill the middle */
  .cap { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #6b7890; margin-top: 14px; padding: 0 4px; }
</style></head><body>
  <div>
    <div class="eyebrow">A course you can watch</div>
    <div class="brand">ag<span>·glass</span></div>
    <h1>See <span>inside</span> a neural network</h1>
    <p>Every number a network computes, and every gradient it sends back, drawn live in your browser.</p>
    <div class="tags"><b>17 lessons</b><b>live training</b><b>no install</b></div>
  </div>
  <div class="pic">
    <div class="crop"><img src="data:image/png;base64,${png}"></div>
    <div class="cap"><span>Two spirals · a real network</span><span>trained live</span></div>
  </div>
</body></html>`;
}
