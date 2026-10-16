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
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; }
  body { width: 1200px; height: 630px; font-family: Inter, sans-serif; background: #f6f6f8; color: #15151a;
         display: grid; grid-template-columns: 1fr 470px; gap: 56px; align-items: center; padding: 0 72px;
         -webkit-font-smoothing: antialiased; position: relative; overflow: hidden; }
  body::before { content: ''; position: absolute; inset: 0 0 auto 0; height: 8px; background: #5e3c99; }
  .brand { font-weight: 700; font-size: 30px; letter-spacing: -0.03em; }
  .brand span { color: #5e3c99; font-weight: 500; }
  h1 { font-size: 64px; line-height: 1.04; letter-spacing: -0.035em; font-weight: 700; margin: 34px 0 24px; }
  h1 span { color: #5e3c99; }
  p { font-size: 25px; line-height: 1.45; color: #4b4b57; max-width: 540px; }
  .tags { display: flex; gap: 10px; margin-top: 34px; }
  .tags b { font-weight: 500; font-size: 18px; color: #5e3c99; background: #f1ecf9; padding: 7px 14px; border-radius: 999px; }
  .pic { background: #fff; border: 1px solid #e8e8ee; border-radius: 22px; padding: 20px;
         box-shadow: 0 20px 50px rgba(40, 30, 80, 0.10), 0 2px 6px rgba(40, 30, 80, 0.05); }
  .crop { overflow: hidden; border-radius: 12px; }
  .crop img { display: block; width: 100%; transform: scale(1.28); }   /* the spirals fill the middle */
  .cap { display: flex; justify-content: space-between; font-size: 16px; color: #8b8b98; margin-top: 14px; padding: 0 4px; }
</style></head><body>
  <div>
    <div class="brand">ag<span>·glass</span></div>
    <h1>See <span>inside</span> a neural network</h1>
    <p>Every number a network computes, and every gradient it sends back, drawn live in your browser.</p>
    <div class="tags"><b>17 lessons</b><b>live training</b><b>no install</b></div>
  </div>
  <div class="pic">
    <div class="crop"><img src="data:image/png;base64,${png}"></div>
    <div class="cap"><span>A real network separating two spirals</span><span>trained live</span></div>
  </div>
</body></html>`;
}
