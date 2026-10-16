// How long the page stops answering while each training lab trains, on an
// emulated phone. A timer that should tick every 20 ms measures every gap;
// the longest gap is how long a tap would wait. Fails when a lab freezes the
// page for long or trains slower than its speed. Chromium emulates a Pixel 7
// with its GPU; WebKit here draws without one, so its numbers are reported,
// not judged.
//
//   npm run build && node scripts/jank.ts [lab,lab,...]

import { spawn, spawnSync } from 'node:child_process';
import { chromium, devices, webkit } from 'playwright';
import { PRESETS } from '../src/graph/presets.ts';

const PORT = 4187;
const base = `http://localhost:${PORT}/`;
const win = process.platform === 'win32';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });

/** Longest acceptable wait for a tap, on the Chromium phone. */
const WORST_MS = 400;
const labs = process.argv[2]?.split(',') ?? PRESETS.filter((p) => p.mode === 'train').map((p) => p.id);
let failed = 0;

try {
  for (let i = 0; i < 40; ++i) { try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ } await sleep(250); }
  for (const [type, device, judged] of [[chromium, 'Pixel 7', true], [webkit, 'iPhone 13', false]] as const) {
    const browser = await type.launch();
    const ctx = await browser.newContext({ ...devices[device] });
    await ctx.addInitScript(() => { try { sessionStorage.setItem('ag-splash-seen', '1'); } catch { /* fine */ } });
    const page = await ctx.newPage();
    for (const id of labs) {
      const preset = PRESETS.find((p) => p.id === id)!;
      await page.goto(`${base}#preset=${id}`);
      await page.waitForSelector('.app[data-engine="ready"]');
      await page.waitForSelector('.train-view');
      await sleep(1500);
      const count = async () => Number(/(\d+)/.exec(await page.locator('.dock .counter').innerText())?.[1] ?? 0);
      await page.locator('.dock').getByRole('button', { name: 'Run' }).click();
      await sleep(1000);
      const before = await count();
      const gaps = await page.evaluate(() => new Promise<number[]>((done) => {
        const gaps: number[] = [];
        let last = performance.now();
        const tick = setInterval(() => { const now = performance.now(); gaps.push(now - last - 20); last = now; }, 20);
        setTimeout(() => { clearInterval(tick); done(gaps); }, 3000);
      }));
      const steps = (await count()) - before;
      await page.locator('.dock').getByRole('button', { name: 'Pause' }).click();
      const worst = Math.max(...gaps);
      const blocked = gaps.filter((g) => g > 50).reduce((a, g) => a + g, 0) / 30;
      const speed = preset.speed ?? 10;
      const slow = steps < speed * 3 * 0.6;
      const bad = judged && (worst > WORST_MS || slow);
      if (bad) failed++;
      console.log(`  ${bad ? 'FAIL' : judged ? 'PASS' : 'INFO'}  ${device.padEnd(9)} ${id.padEnd(13)} longest wait ${worst.toFixed(0).padStart(4)} ms, blocked ${blocked.toFixed(0).padStart(3)}%, ${steps} steps in 3 s (speed ${speed})`);
    }
    await browser.close();
  }
} finally {
  if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}
console.log(failed ? `=== ${failed} failed ===` : '=== all passed ===');
process.exit(failed ? 1 : 0);
