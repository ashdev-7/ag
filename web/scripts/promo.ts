// A 15-second promo video, recorded from the real site: it opens with the
// splash, a cursor clicks "Watch a network train", presses Run, turns up the
// speed, and the page settles on the network and its decision boundary while
// it learns; then the address. Frames come from the browser's screencast and
// are encoded to H.264 by ffmpeg (pip install imageio-ffmpeg), the format X,
// LinkedIn, Reddit and Product Hunt accept. No sound.
//
//   npm run build && node scripts/promo.ts out.mp4             1920x1080, a desktop and a cursor
//   npm run build && node scripts/promo.ts out.mp4 vertical    1080x1920, a phone and a finger

import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright';
import type { Locator, Page } from 'playwright';

const out = resolve(process.argv[2] ?? 'promo.mp4');
const vertical = process.argv[3] === 'vertical';
const SECONDS = 15;
// drawn at a page size, captured at 1080p
const W = vertical ? 405 : 1440, H = vertical ? 720 : 810, SCALE = vertical ? 8 / 3 : 4 / 3;
const OUT_W = Math.round(W * SCALE), OUT_H = Math.round(H * SCALE);
const PORT = 4188;
const base = `http://localhost:${PORT}/`;
const win = process.platform === 'win32';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const ffmpeg = execFileSync('python', ['-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())']).toString().trim();
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });

// a visible pointer: the screencast does not draw the real one. On a desktop
// an arrow that follows the mouse; on a phone a fingertip. A soft ring on each press
function pointer(finger: boolean) {
  addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div');
    c.id = 'promo-cursor';
    c.innerHTML = finger
      ? '<div style="width:34px;height:34px;border-radius:50%;background:rgba(29,29,31,.22);border:2px solid rgba(255,255,255,.85);box-shadow:0 2px 8px rgba(0,0,0,.18)"></div>'
      : '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21 L8.2 16.4 L11.6 24 L15 22.5 L11.7 15 L19 15 Z" fill="#1d1d1f" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, { position: 'fixed', left: '0', top: '0', zIndex: '99999', pointerEvents: 'none', opacity: '0', transition: 'opacity .3s',
      filter: finger ? 'none' : 'drop-shadow(0 2px 3px rgba(0,0,0,.25))' });
    document.body.appendChild(c);
    const [dx, dy] = finger ? [17, 17] : [3, 2];
    addEventListener('mousemove', (e) => { c.style.transform = `translate(${e.clientX - dx}px, ${e.clientY - dy}px)`; }, true);
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

let mouse = vertical ? { x: W * 0.7, y: H * 0.62 } : { x: W * 0.6, y: H * 0.36 };   // starts away from anything that lights up
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Glides the pointer to (x, y) over `ms`, the way a hand moves: fast in the middle. */
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

/** Scrolls the page smoothly to `y` over `ms`, as a trackpad or a flick would. */
const scrollTo = (page: Page, y: number, ms: number) => page.evaluate(([y1, ms]) => new Promise<void>((done) => {
  const y0 = scrollY, t0 = performance.now();
  const f = (now: number) => {
    const t = Math.min(1, (now - t0) / ms);
    window.scrollTo(0, y0 + (y1 - y0) * (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2));
    if (t < 1) requestAnimationFrame(f); else done();
  };
  requestAnimationFrame(f);
}), [y, ms] as const);

/** Where an element sits on the page (not the screen). */
const pageY = (l: Locator) => l.evaluate((e) => e.getBoundingClientRect().top + scrollY);

const frames: { data: Buffer; t: number }[] = [];
const now = () => Date.now() / 1000;
const press = async (page: Page) => { await page.mouse.down(); await sleep(90); await page.mouse.up(); };

try {
  for (let i = 0; i < 40; ++i) { try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ } await sleep(250); }
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: SCALE, isMobile: vertical });
  await ctx.addInitScript(pointer, vertical);
  const page = await ctx.newPage();
  const cdp = await ctx.newCDPSession(page);
  cdp.on('Page.screencastFrame', (f) => {
    frames.push({ data: Buffer.from(f.data, 'base64'), t: f.metadata.timestamp ?? now() });
    void cdp.send('Page.screencastFrameAck', { sessionId: f.sessionId });
  });
  await page.goto('about:blank');
  await cdp.send('Page.startScreencast', { format: 'png', maxWidth: OUT_W, maxHeight: OUT_H, everyNthFrame: 1 });
  const start = Date.now();
  const at = async (s: number) => { const wait = start + s * 1000 - Date.now(); if (wait > 0) await sleep(wait); };
  const show = (on: boolean) => page.evaluate((on) => { document.getElementById('promo-cursor')!.style.opacity = on ? '1' : '0'; }, on);

  // 0 s: the site opens on its splash
  await page.goto(base);
  await page.waitForSelector('.splash');
  await page.mouse.move(mouse.x, mouse.y);
  // 3 s: the home page; the pointer appears, once the splash has fully gone
  await at(3.0);
  await page.waitForFunction(() => !document.querySelector('.splash'));
  await show(true);
  const watch = page.getByRole('link', { name: 'Watch a network train' });
  const w = await centre(watch);
  await glide(page, w.x, w.y, 1200);
  await at(4.5);
  await press(page);
  // the hero's animation can still be settling the layout: if the press missed, click the link itself
  if (!(await page.waitForURL(/preset=moons/, { timeout: 1500 }).then(() => true, () => false))) await watch.click();
  await page.waitForFunction(() => document.querySelector('.app')?.getAttribute('data-engine') === 'ready' && document.querySelector('.train-view'));

  // the lab: press Run (on a phone, in the bar along the bottom)
  const run = page.locator(vertical ? '.dock' : '.toolbar').getByRole('button', { name: 'Run' });
  await run.waitFor();
  const r = await centre(run);
  await glide(page, r.x, r.y, 750);
  await at(6.1);
  await press(page);
  // turn the speed up: drag the slider from 20 to 100 steps a second
  const slider = (await page.getByLabel('Training speed').boundingBox())!;
  await glide(page, slider.x + slider.width * (4 / 10), slider.y + slider.height / 2, 550);
  await page.mouse.down();
  await glide(page, slider.x + slider.width * (6 / 10), slider.y + slider.height / 2, 600);
  await page.mouse.up();

  // the network and its decision boundary, while it learns
  await at(7.9);
  if (vertical) {
    // a phone shows one panel at a time: the boundary bending, then the network that bent it
    await show(false);
    await scrollTo(page, (await pageY(page.locator('.train-grid > *').nth(1))) - 64, 1300);
    await at(11.3);
    await scrollTo(page, (await pageY(page.locator('.train-grid > *').nth(0))) - 64, 1100);
  } else {
    void scrollTo(page, (await pageY(page.locator('.train-grid'))) - 70, 1300);
    await glide(page, W * 0.97, H * 0.5, 1300);   // out of the way, then gone
    await show(false);
  }

  // 13 s: the address, over the trained network
  await at(13.0);
  await page.evaluate((vertical) => {
    const k = vertical ? 0.62 : 1;
    const card = document.createElement('div');
    card.innerHTML = `<div style="font:700 ${64 * k}px Inter,system-ui,sans-serif;letter-spacing:-0.02em;color:#1d1d1f">ag<span style="color:#5e3c99">·glass</span></div>`
      + `<div style="font:500 ${26 * k}px Inter,system-ui,sans-serif;color:#3d3d48;margin-top:10px">See inside a neural network</div>`
      + `<div style="font:600 ${24 * k}px Inter,system-ui,sans-serif;color:#5e3c99;margin-top:${28 * k}px;padding:${10 * k}px ${22 * k}px;border:1.5px solid rgba(94,60,153,.35);border-radius:999px;background:#fff">ag-glass-box.vercel.app</div>`;
    Object.assign(card.style, { position: 'fixed', inset: '0', zIndex: '99997', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(247,247,250,.9)', backdropFilter: 'blur(3px)', opacity: '0', transition: 'opacity .6s ease' });
    document.body.appendChild(card);
    requestAnimationFrame(() => { card.style.opacity = '1'; });
  }, vertical);
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
  '-vf', `fps=30,scale=${OUT_W}:${OUT_H}:flags=lanczos,format=yuv420p`, '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
  '-movflags', '+faststart', '-t', String(SECONDS), out], { cwd: dir, stdio: 'inherit' });
rmSync(dir, { recursive: true, force: true });
console.log(`${kept.length} frames -> ${out}`);
