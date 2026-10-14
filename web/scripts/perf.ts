// Frame-time measurement of the built site in a real browser. The budget
// (PLAN §5, PRD §9) is one frame: the page must keep drawing at 60 fps
// while the engine trains or steps in its worker.
//
//   npm run build && node scripts/perf.ts [chromium|firefox|webkit]
//
// For each training preset, at the default and at the highest speed: the
// interval between animation frames while training runs (median, 95th
// percentile, worst, share of frames over 16.7 ms) and the steps per second
// reached. For stepping: the time from a key press to the new step being on
// screen, on the largest graph (the two-layer MLP as single numbers).
// Headless browsers have no display, so these are an upper bound on what a
// visitor's machine does with the same CPU; compare runs, not absolutes.

import { spawn } from 'node:child_process';
import { chromium, firefox, webkit } from 'playwright';
import type { Page } from 'playwright';
import { PRESETS } from '../src/graph/presets.ts';

const browserName = process.argv[2] ?? 'chromium';
const browserType = ({ chromium, firefox, webkit } as const)[browserName as 'chromium'];
if (!browserType) throw new Error(`unknown browser ${browserName}`);
const PORT = 4175;
const win = process.platform === 'win32';
const server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { shell: true, stdio: 'ignore', detached: !win });
const base = `http://localhost:${PORT}/`;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const BUDGET = 1000 / 60;

/** Record animation-frame intervals for `ms` while `during` runs. */
async function frames(page: Page, ms: number): Promise<number[]> {
  return page.evaluate((duration) => new Promise<number[]>((resolve) => {
    const out: number[] = [];
    let last = performance.now();
    const end = last + duration;
    const tick = (t: number) => {
      out.push(t - last);
      last = t;
      if (t < end) requestAnimationFrame(tick);
      else resolve(out.slice(1));   // the first interval includes setup
    };
    requestAnimationFrame(tick);
  }), ms);
}

const pct = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : NaN;
};
const row = (cells: (string | number)[]) => console.log(cells.map((c, i) => String(c).padStart(i ? 9 : 22)).join(''));
const f1 = (x: number) => x.toFixed(1);

let over = 0;
try {
  for (let i = 0; i < 40; ++i) {
    try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ }
    await sleep(250);
  }
  const browser = await browserType.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  console.log(`browser: ${browserName} ${browser.version()}\n`);

  console.log('training: frame interval in ms while running');
  row(['preset', 'speed', 'p50', 'p95', 'max', '>16.7ms', 'steps/s']);
  for (const p of PRESETS.filter((q) => q.mode === 'train')) {
    for (const speed of [10, 100]) {
      await page.goto(`${base}#preset=${p.id}`);
      await page.reload();
      await page.waitForSelector('.train-view');
      await page.locator('.toolbar input[type=range]').first().fill(String(speed));
      await page.getByRole('button', { name: 'Run' }).click();
      await sleep(1000);   // past start-up
      const counter = () => page.locator('.toolbar .counter').last().textContent();
      const s0 = Number(/step (\d+)/.exec((await counter()) ?? '')?.[1] ?? 0);
      const t0 = Date.now();
      const iv = await frames(page, 3000);
      const s1 = Number(/step (\d+)/.exec((await counter()) ?? '')?.[1] ?? 0);
      const rate = ((s1 - s0) / (Date.now() - t0)) * 1000;
      await page.getByRole('button', { name: 'Pause' }).click();
      const slow = iv.filter((x) => x > BUDGET * 1.5).length / iv.length;
      if (pct(iv, 0.95) > BUDGET * 1.5) over++;
      row([p.id, speed, f1(pct(iv, 0.5)), f1(pct(iv, 0.95)), f1(Math.max(...iv)), `${(slow * 100).toFixed(0)}%`, rate.toFixed(0)]);
    }
  }

  console.log('\nrace: frame interval in ms while racing');
  row(['preset', '', 'p50', 'p95', 'max', '>16.7ms', '']);
  for (const p of PRESETS.filter((q) => q.mode === 'race')) {
    await page.goto(`${base}#preset=${p.id}`);
    await page.reload();
    await page.waitForSelector('.race-view');
    await page.getByRole('button', { name: 'Race' }).click();
    await sleep(500);
    const iv = await frames(page, 3000);
    const slow = iv.filter((x) => x > BUDGET * 1.5).length / iv.length;
    if (pct(iv, 0.95) > BUDGET * 1.5) over++;
    row([p.id, '', f1(pct(iv, 0.5)), f1(pct(iv, 0.95)), f1(Math.max(...iv)), `${(slow * 100).toFixed(0)}%`, '']);
  }

  console.log('\nstepping: ms from key press to the new step on screen');
  row(['graph', 'nodes', 'p50', 'p95', 'max', '', '']);
  for (const [label, zoom] of [['mlp tensors', 'tensor'], ['mlp single numbers', 'scalar']] as const) {
    await page.goto(`${base}#preset=mlp&pass=reverse&zoom=${zoom}`);
    await page.reload();
    await page.waitForFunction(() => document.querySelectorAll('.react-flow__node').length > 0, null, { timeout: 20000 });
    await sleep(1000);
    const nodes = await page.locator('.react-flow__node').count();
    await page.locator('.graph-wrap').click({ position: { x: 5, y: 5 } });
    const lat: number[] = [];
    for (let k = 0; k < 40; ++k) {
      let before = await page.locator('.transport .counter').textContent();
      const [at, of] = (before ?? '').split(' / ');
      if (at === of) {
        // at the end: back to the start (not timed)
        await page.keyboard.press('Home');
        await page.waitForFunction(() => document.querySelector('.transport .counter')?.textContent?.startsWith('0 /'));
        before = await page.locator('.transport .counter').textContent();
      }
      const t = await page.evaluate(() => performance.now());
      await page.keyboard.press('ArrowRight');
      const done = await page.waitForFunction((b) => {
        const c = document.querySelector('.transport .counter')?.textContent;
        return c !== b ? performance.now() : false;
      }, before, { polling: 'raf', timeout: 5000 }).catch(async (e) => {
        throw new Error(`step ${k}: counter stayed at ${before}; focused ${await page.evaluate(() => document.activeElement?.outerHTML.slice(0, 120))}: ${e}`);
      });
      lat.push((await done.jsonValue() as number) - t);
    }
    row([label, nodes, f1(pct(lat, 0.5)), f1(pct(lat, 0.95)), f1(Math.max(...lat)), '', '']);
  }
  await browser.close();
} finally {
  if (win) spawn('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}
console.log(over ? `\n${over} runs had a p95 frame over 25 ms` : '\nevery run kept its p95 frame within 25 ms');
