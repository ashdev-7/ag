// A 15-second promo video, recorded from the real site: it opens with the
// splash, a cursor clicks "Watch a network train", presses Run, turns up the
// speed, and the page settles on the network and its decision boundary while
// it learns. Frames come from the browser's screencast at 1920x1080 and are
// encoded to H.264 by ffmpeg (pip install imageio-ffmpeg), the format X,
// LinkedIn, Reddit and Product Hunt accept.
//
//   npm run build && node scripts/promo.ts out.mp4

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright';
import type { Locator, Page } from 'playwright';

const out = resolve(process.argv[2] ?? 'promo.mp4');
const SECONDS = 15;
const W = 1440, H = 810, SCALE = 4 / 3;   // drawn at 1440x810, captured at 1920x1080
const PORT = 4188;
const base = `http://localhost:${PORT}/`;
const win = process.platform === 'win32';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const ffmpeg = execFileSync('python', ['-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())']).toString().trim();
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });

// a visible cursor: the screencast does not draw the real one. An arrow that
// follows the mouse, with a soft ring on each click
function cursor() {
  addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div');
    c.id = 'promo-cursor';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21 L8.2 16.4 L11.6 24 L15 22.5 L11.7 15 L19 15 Z" fill="#1d1d1f" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, { position: 'fixed', left: '0', top: '0', zIndex: '99999', pointerEvents: 'none', opacity: '0', transition: 'opacity .3s', filter: 'drop-shadow(0 2px 3px rgba(0,0,0,.25))' });
    document.body.appendChild(c);
    addEventListener('mousemove', (e) => { c.style.transform = `translate(${e.clientX - 3}px, ${e.clientY - 2}px)`; }, true);
    addEventListener('mousedown', (e) => {
      const r = document.createElement('div');
      Object.assign(r.style, { position: 'fixed', left: `${e.clientX - 18}px`, top: `${e.clientY - 18}px`, width: '36px', height: '36px', borderRadius: '50%',
        background: 'rgba(94,60,153,.28)', zIndex: '99998', pointerEvents: 'none', transform: 'scale(.3)', transition: 'transform .45s ease-out, opacity .45s ease-out' });
      document.body.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(1.4)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 500);
    }, true);
  });
}

let mouse = { x: W * 0.6, y: H * 0.36 };   // starts over the hero, away from anything that lights up
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Glides the cursor to (x, y) over `ms`, the way a hand moves: fast in the middle. */
async function glide(page: Page, x: number, y: number, ms: number) {
  const from = { ...mouse }, t0 = Date.now();
  for (;;) {
    const t = Math.min(1, (Date.now() - t0) / ms), k = ease(t);
    await page.mouse.move(from.x + (x - from.x) * k, from.y + (y - from.y) * k);
    if (t >= 1) break;
    await sleep(12);
  }
  mouse = { x, y };
}

async function centre(l: Locator) {
  const b = (await l.boundingBox())!;
  return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
}

/** Scrolls the page smoothly by `dy` over `ms`, as a trackpad would. */
const scrollBy = (page: Page, dy: number, ms: number) => page.evaluate(([dy, ms]) => new Promise<void>((done) => {
  const y0 = scrollY, t0 = performance.now();
  const f = (now: number) => {
    const t = Math.min(1, (now - t0) / ms);
    scrollTo(0, y0 + dy * (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2));
    if (t < 1) requestAnimationFrame(f); else done();
  };
  requestAnimationFrame(f);
}), [dy, ms] as const);

const frames: { data: Buffer; t: number }[] = [];
try {
  for (let i = 0; i < 40; ++i) { try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ } await sleep(250); }
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: SCALE });
  await ctx.addInitScript(cursor);
  const page = await ctx.newPage();
  const cdp = await ctx.newCDPSession(page);
  cdp.on('Page.screencastFrame', (f) => {
    frames.push({ data: Buffer.from(f.data, 'base64'), t: f.metadata.timestamp ?? Date.now() / 1000 });
    void cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId });
  });
  await page.goto('about:blank');
  await cdp.send('Page.startScreencast', { format: 'png', maxWidth: Math.round(W * SCALE), maxHeight: Math.round(H * SCALE), everyNthFrame: 1 });
  const start = Date.now();
  const at = async (s: number) => { const wait = start + s * 1000 - Date.now(); if (wait > 0) await sleep(wait); };

  // 0 s: the site opens on its splash
  await page.goto(base);
  await page.mouse.move(mouse.x, mouse.y);
  // 3 s: the home page; the cursor appears and goes to "Watch a network train"
  await at(3.0);
  await page.waitForFunction(() => !document.querySelector('.splash'));   // only once the splash has fully gone
  await page.evaluate(() => { document.getElementById('promo-cursor')!.style.opacity = '1'; });
  const watch = page.getByRole('link', { name: 'Watch a network train' });
  const w = await centre(watch);
  await glide(page, w.x, w.y, 1200);
  await at(4.5);
  await page.mouse.down(); await sleep(90); await page.mouse.up();
  // the hero's animation can still be settling the layout: if the click missed, click the link itself
  if (!(await page.waitForURL(/preset=moons/, { timeout: 1500 }).then(() => true, () => false))) await watch.click();
  await page.waitForURL(/preset=moons/);
  // the lab: press Run
  await page.waitForFunction(() => document.querySelector('.app')?.getAttribute('data-engine') === 'ready' && document.querySelector('.train-view'));
  const run = page.locator('.toolbar').getByRole('button', { name: 'Run' });
  await run.waitFor();
  const r = await centre(run);
  await glide(page, r.x, r.y, 750);
  await at(6.1);
  await page.mouse.down(); await sleep(90); await page.mouse.up();
  // turn the speed up: drag the slider most of the way
  const slider = (await page.getByLabel('Training speed').boundingBox())!;
  const knobX = slider.x + slider.width * (4 / 10);   // 20 steps a second sits at 4 of 10
  await glide(page, knobX, slider.y + slider.height / 2, 550);
  await page.mouse.down();
  await glide(page, slider.x + slider.width * (6 / 10), slider.y + slider.height / 2, 600);   // 100 steps a second
  await page.mouse.up();
  // the network and its decision boundary, while it learns
  await at(7.9);
  const grid = (await page.locator('.train-grid').boundingBox())!;
  void scrollBy(page, grid.y - 70, 1300);
  await glide(page, W * 0.97, H * 0.5, 1300);   // out of the way, then gone
  await page.evaluate(() => { document.getElementById('promo-cursor')!.style.opacity = '0'; });
  // 13 s: the address, over the trained network
  await at(13.0);
  await page.evaluate(() => {
    const card = document.createElement('div');
    card.innerHTML = '<div style="font:700 64px Inter,system-ui,sans-serif;letter-spacing:-0.02em;color:#1d1d1f">ag<span style="color:#5e3c99">·glass</span></div>'
      + '<div style="font:500 26px Inter,system-ui,sans-serif;color:#3d3d48;margin-top:10px">See inside a neural network</div>'
      + '<div style="font:600 24px Inter,system-ui,sans-serif;color:#5e3c99;margin-top:28px;padding:10px 22px;border:1.5px solid rgba(94,60,153,.35);border-radius:999px;background:#fff">ag-glass-box.vercel.app</div>';
    Object.assign(card.style, { position: 'fixed', inset: '0', zIndex: '99997', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(247,247,250,.9)', backdropFilter: 'blur(3px)', opacity: '0', transition: 'opacity .6s ease' });
    document.body.appendChild(card);
    requestAnimationFrame(() => { card.style.opacity = '1'; });
  });
  await at(SECONDS + 0.2);
  await cdp.send('Page.stopScreencast');
  await browser.close();
} finally {
  if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}

// each frame lasts until the next one arrived; ffmpeg samples that at 30 fps
const dir = join(tmpdir(), `ag-promo-${Date.now()}`);
mkdirSync(dir, { recursive: true });
const t0 = frames[0].t;
const kept = frames.filter((f) => f.t - t0 <= SECONDS);
let list = '';
kept.forEach((f, i) => {
  const name = `f${String(i).padStart(5, '0')}.png`;
  writeFileSync(join(dir, name), f.data);
  const next = kept[i + 1]?.t ?? t0 + SECONDS;
  list += `file '${name}'\nduration ${Math.max(0.001, next - f.t).toFixed(4)}\n`;
});
list += `file 'f${String(kept.length - 1).padStart(5, '0')}.png'\n`;
writeFileSync(join(dir, 'list.txt'), list);
execFileSync(ffmpeg, ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', join(dir, 'list.txt'),
  '-vf', 'fps=30,scale=1920:1080:flags=lanczos,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
  '-movflags', '+faststart', '-t', String(SECONDS), out], { cwd: dir, stdio: 'inherit' });
rmSync(dir, { recursive: true, force: true });
console.log(`${kept.length} frames -> ${out}`);
