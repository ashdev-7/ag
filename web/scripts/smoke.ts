// Browser smoke test of the built site: every preset loads, steps, trains or
// races, with no console errors. Screenshots go to the directory given.
//
//   npm run build && node scripts/smoke.ts [screenshot-dir] [chromium|firefox|webkit]
//
// webkit is the engine behind Safari; Playwright's build is close to Safari,
// not identical to it.
//
// Serves dist/ itself, so it tests exactly what would be deployed.

import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium, firefox, webkit } from 'playwright';
import { PRESETS } from '../src/graph/presets.ts';

const browserName = process.argv[3] ?? 'chromium';
const browserType = ({ chromium, firefox, webkit } as const)[browserName as 'chromium'];
if (!browserType) throw new Error(`unknown browser ${browserName}`);
const shots = join(process.argv[2] ?? 'smoke-shots', browserName);
mkdirSync(shots, { recursive: true });
const PORT = 4173;
const win = process.platform === 'win32';
// its own process group on unix, so the whole tree can be stopped at the end
const server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { shell: true, stdio: 'ignore', detached: !win });
const base = `http://localhost:${PORT}/`;

let failed = 0;
const check = (ok: boolean, label: string) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) failed++;
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

try {
  for (let i = 0; i < 40; ++i) {
    try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ }
    await sleep(250);
  }
  const browser = await browserType.launch();
  console.log(`browser: ${browserName} ${browser.version()}`);
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto(base);
  await page.waitForSelector('.cards');
  check((await page.locator('.card').count()) === PRESETS.length, 'landing lists every preset');
  await page.screenshot({ path: join(shots, '00-landing.png') });

  for (const p of PRESETS) {
    errors.length = 0;
    await page.goto(`${base}#preset=${p.id}`);
    await page.waitForSelector('.engine-ready', { timeout: 20000 });
    if (p.mode === 'step') {
      await page.waitForSelector('.react-flow__node');
      // every edge, drawn: a race in React Flow once left graphs with none
      const wantEdges = p.spec.nodes.reduce((k, n) => k + n.inputs.length, 0);
      await sleep(500);
      const edges = await page.locator('.react-flow__edge-path').count();
      check(edges === wantEdges, `${p.id}: all ${wantEdges} edges drawn (${edges})`);
      for (let k = 0; k < 6; ++k) await page.getByTitle('One step forward').click();
      // the counter follows the engine's reply, a moment after the click
      await page.waitForFunction(() => document.querySelector('.transport .counter')?.textContent?.startsWith('6 /'), null, { timeout: 5000 }).catch(() => {});
      const counter = await page.locator('.transport .counter').textContent();
      check(counter?.startsWith('6 /') ?? false, `${p.id}: six steps forward (${counter})`);
      await page.getByTitle('To the end').click();
      await page.locator('.react-flow__node').first().click();
      await sleep(300);
      if (p.id === 'detach') {
        check(await page.locator('.react-flow__edge.gedge-stopped').count() === 1, 'detach: the edge into detach(a) is drawn as stopped');
        check((await page.locator('.react-flow__node', { hasText: 'a' }).first().textContent())?.includes('∂ 1.00') ?? false,
          'detach: a gets 1, not 4');
      }
      await page.screenshot({ path: join(shots, `${p.id}-tensor.png`) });
      await page.getByText('Reverse vs forward mode').click();
      await page.waitForSelector('.modes table', { timeout: 20000 });
      const match = await page.locator('.modes .ok').count();
      check(match === 1, `${p.id}: forward mode matches reverse mode`);
      await page.getByRole('button', { name: 'Single numbers' }).click();
      // the scalar graph is built in the worker and laid out; give it time
      await page.waitForFunction(() => document.querySelectorAll('.react-flow__node').length > 0, null, { timeout: 15000 }).catch(() => {});
      await sleep(300);
      const scalarNodes = await page.locator('.react-flow__node').count();
      check(scalarNodes > 0, `${p.id}: single-number view has ${scalarNodes} nodes`);
      await page.getByTitle('To the end').click();
      await sleep(400);
      await page.screenshot({ path: join(shots, `${p.id}-scalar.png`) });
      // forward mode, then switching zoom: the seed must follow the graph
      // without a false error
      await page.getByRole('button', { name: 'Forward mode', exact: true }).click();
      await sleep(400);
      await page.getByRole('button', { name: 'Tensors' }).click();
      await sleep(600);
      await page.getByRole('button', { name: 'Single numbers' }).click();
      await sleep(800);
      const banner = await page.locator('.banner-error').count();
      check(banner === 0, `${p.id}: switching zoom in forward mode shows no error`);
      await page.getByRole('button', { name: 'Backward', exact: true }).click();
      await page.getByRole('button', { name: 'Tensors', exact: true }).click();
    } else if (p.mode === 'train') {
      await page.waitForSelector('.train-view');
      await page.getByRole('button', { name: 'Run' }).click();
      await sleep(3000);
      await page.getByRole('button', { name: 'Pause' }).click();
      const text = await page.locator('.toolbar .counter').last().textContent();
      const step = Number(/step (\d+)/.exec(text ?? '')?.[1] ?? 0);
      check(step > 100, `${p.id}: trained ${step} steps in 3 s`);
      await page.locator('.react-flow__node', { hasText: 'W1' }).first().click();
      await sleep(500);
      await page.screenshot({ path: join(shots, `${p.id}.png`), fullPage: true });
    } else {
      await page.waitForSelector('.race-view');
      await page.getByRole('button', { name: 'Race' }).click();
      await sleep(3000);
      await page.screenshot({ path: join(shots, `${p.id}.png`) });
      check(true, `${p.id}: raced`);
    }
    check(errors.length === 0, `${p.id}: no console errors${errors.length ? ': ' + errors.slice(0, 2).join(' | ') : ''}`);
  }

  // two optimizers side by side
  errors.length = 0;
  await page.goto(`${base}#preset=moons&opt=sgd&lr=0.3&cmp=adam`);
  await page.waitForSelector('.engine-ready');
  await page.waitForSelector('.train-view');
  check((await page.locator('.optim').count()) === 2, 'compare: two optimizers from the link');
  await page.getByRole('button', { name: 'Run' }).click();
  await sleep(3000);
  await page.getByRole('button', { name: 'Pause' }).click();
  await page.screenshot({ path: join(shots, 'compare.png'), fullPage: true });
  check(errors.length === 0, `compare: no console errors${errors.length ? ': ' + errors[0] : ''}`);

  // a link to a point in a pass opens there, with the backward order shown
  errors.length = 0;
  await page.goto(`${base}#preset=expression&pass=reverse&i=7`);
  await page.waitForSelector('.react-flow__node');
  await page.waitForFunction(() => document.querySelector('.transport .counter')?.textContent?.startsWith('7 /'), null, { timeout: 10000 }).catch(() => {});
  check((await page.locator('.transport .counter').textContent())?.startsWith('7 /') ?? false, 'link: opens at step 7');
  const queue = await page.locator('.queue li').allTextContents();
  check(queue.join('|') === 'ddone|cready|awaiting 1 of 2|bwaiting', `link: backward order at step 7 (${queue.join(', ')})`);
  check(errors.length === 0, `link: no console errors${errors.length ? ': ' + errors[0] : ''}`);

  // the stepper from the keyboard alone
  const counter = () => page.locator('.transport .counter').textContent();
  const pressAndRead = async (key: string) => {
    const before = await counter();
    await page.keyboard.press(key);
    await page.waitForFunction((b) => document.querySelector('.transport .counter')?.textContent !== b, before, { timeout: 3000 }).catch(() => {});
    return counter();
  };
  await page.locator('.graph-wrap').click({ position: { x: 5, y: 5 } });   // focus the page, not a control
  check((await pressAndRead('ArrowRight'))?.startsWith('8 /') ?? false, 'keys: → steps forward');
  check((await pressAndRead('ArrowLeft'))?.startsWith('7 /') ?? false, 'keys: ← steps back');
  check((await pressAndRead('End'))?.startsWith('9 /') ?? false, 'keys: End jumps to the end');
  check((await pressAndRead('Home'))?.startsWith('0 /') ?? false, 'keys: Home jumps to the start');
  await page.keyboard.press(' ');
  await sleep(1500);
  const played = Number((await counter())?.split(' ')[0]);
  await page.keyboard.press(' ');
  check(played > 0, `keys: space plays (reached step ${played})`);

  // a phone-sized screen
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${base}#preset=expression`);
  await page.waitForSelector('.react-flow__node');
  await page.screenshot({ path: join(shots, 'phone.png'), fullPage: true });

  await browser.close();
} finally {
  if (win) spawn('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}
console.log(`=== ${failed ? `${failed} failed` : 'all passed'} ===`);
process.exit(failed ? 1 : 0);
