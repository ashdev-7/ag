// Browser smoke test of the built site: every preset loads, steps, trains or
// races, with no console errors. Screenshots go to the directory given.
//
//   npm run build && node scripts/smoke.ts [screenshot-dir] [chromium|firefox|webkit]
//
// webkit is the engine behind Safari; Playwright's build is close to Safari,
// not identical to it.
//
// Serves dist/ itself, so it tests exactly what would be deployed.

import { spawn, spawnSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium, firefox, webkit } from 'playwright';
import type { Page } from 'playwright';
import { PRESETS } from '../src/graph/presets.ts';
import { SPEEDS } from '../src/views/speed.ts';

const browserName = process.argv[3] ?? 'chromium';
const browserType = ({ chromium, firefox, webkit } as const)[browserName as 'chromium'];
if (!browserType) throw new Error(`unknown browser ${browserName}`);
const shots = join(process.argv[2] ?? 'smoke-shots', browserName);
mkdirSync(shots, { recursive: true });
const PORT = 4173;
const win = process.platform === 'win32';
// its own process group on unix, so the whole tree can be stopped at the end
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });
const base = `http://localhost:${PORT}/`;

let failed = 0;
const check = (ok: boolean, label: string) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
  if (!ok) failed++;
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let page: Page;

/** Run, wait, pause; the training step reached and the seconds it ran. */
async function trainFor(ms: number): Promise<{ step: number; secs: number }> {
  const read = async () => Number(/step (\d+)/.exec((await page.locator('.toolbar .step-count').textContent()) ?? '')?.[1] ?? 0);
  const before = await read();
  const t0 = Date.now();
  await page.getByRole('button', { name: 'Run' }).click();
  await sleep(ms);
  await page.getByRole('button', { name: 'Pause' }).click();
  const secs = (Date.now() - t0) / 1000;
  await sleep(300);   // the last frame in flight
  return { step: (await read()) - before, secs };
}

const setSpeed = (stepsPerSecond: number) => page.getByLabel('Training speed').fill(String(SPEEDS.indexOf(stepsPerSecond)));

try {
  for (let i = 0; i < 40; ++i) {
    try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ }
    await sleep(250);
  }
  const browser = await browserType.launch();
  console.log(`browser: ${browserName} ${browser.version()}`);
  page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto(base);
  await page.waitForSelector('.cards');
  check((await page.locator('.card').count()) === PRESETS.length, 'landing lists every preset');
  await page.screenshot({ path: join(shots, '00-landing.png') });

  // every lesson opens, draws its figures, and links to its labs
  // (the lesson files are .tsx, which node cannot load here, so the list comes from the page)
  const lessonIds = (await page.locator('.lesson-list a').evaluateAll((as) => as.map((a) => a.getAttribute('href') ?? '')))
    .map((h) => h.replace('#lesson=', ''));
  check(lessonIds.length >= 17, `landing lists ${lessonIds.length} lessons`);
  for (const id of lessonIds) {
    errors.length = 0;
    await page.goto(`${base}#lesson=${id}`);
    await page.waitForSelector('.lesson h1');
    const widgets = await page.locator('.widget canvas').count();
    const labs = await page.locator('.labs a.button').count();
    check(labs >= 1 && errors.length === 0, `lesson ${id}: ${labs} lab links, ${widgets} figures, ${errors.length ? errors[0] : 'no console errors'}`);
  }
  await page.goto(`${base}#lesson=derivatives`);
  await page.waitForSelector('.widget canvas');
  await page.screenshot({ path: join(shots, '00-lesson.png'), fullPage: true });

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
      await page.getByRole('button', { name: 'Reverse vs forward mode' }).click();
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
      // trains at the speed the lab opens at: never faster, not far slower
      const { step, secs } = await trainFor(3000);
      const want = (p.speed ?? 10) * secs;
      check(step <= want * 1.25 + 2 && step >= want * 0.5,
        `${p.id}: trains at its ${p.speed ?? 10} steps/s (${step} steps in ${secs.toFixed(1)} s)`);
      // the neuron diagram: hovering the first hidden neuron explains it
      const net = await page.locator('.neurons canvas').boundingBox();
      let explained = false, said = '';
      for (let y = 0.03; y < 0.95 && !explained && net; y += 0.006) {
        await page.mouse.move(net.x + 24 + (net.width - 122) / (p.layers.length) + 4, net.y + net.height * y);
        said = (await page.locator('.neurons .story').textContent()) ?? '';
        explained = /^layer \d+ \(\w+\), neuron \d+: it adds up/.test(said);
      }
      check(explained, `${p.id}: hovering a neuron explains it${explained ? '' : ` (last said: ${said.slice(0, 60)})`}`);
      await page.getByRole('button', { name: 'Calculations' }).click();
      await page.waitForSelector('.react-flow__node');
      await page.locator('.react-flow__node', { hasText: 'W1' }).first().click();
      await sleep(500);
      await page.screenshot({ path: join(shots, `${p.id}.png`), fullPage: true });
    } else {
      await page.waitForSelector('.race-view');
      const t0 = Date.now();
      await page.getByRole('button', { name: 'Race' }).click();
      await sleep(3000);
      const raced = Number(/step (\d+)/.exec((await page.locator('.step-count').textContent()) ?? '')?.[1] ?? 0);
      const want = p.speed! * (Date.now() - t0) / 1000;
      check(raced <= want * 1.25 + 2 && raced >= want * 0.5, `${p.id}: races at its ${p.speed} steps/s (step ${raced} after 3 s)`);
      await page.screenshot({ path: join(shots, `${p.id}.png`) });
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

  // slow speeds are really slow: steps per second, not per display frame
  errors.length = 0;
  await page.goto(`${base}#preset=moons`);
  await page.waitForSelector('.train-view');
  await page.waitForSelector('.engine-ready');
  for (const [sps, ms] of [[1, 4000], [5, 4000]] as const) {
    await setSpeed(sps);
    const { step, secs } = await trainFor(ms);
    // step k is due at k / speed seconds, the first at once; one more for the
    // delay between the page's clock and the worker's
    check(step <= Math.floor(sps * secs) + 2 && step >= sps * secs * 0.6, `speed: ${sps} steps/s gives ${step} steps in ${secs.toFixed(1)} s`);
  }
  check(errors.length === 0, `speed: no console errors${errors.length ? ': ' + errors[0] : ''}`);

  // the descent lesson: sgd at lr 0.11 on the bowl is marked diverged
  errors.length = 0;
  await page.goto(`${base}#preset=bowl`);
  await page.waitForSelector('.race-view');
  await page.locator('.optim').first().getByLabel('lr').fill('0.11');
  await page.getByRole('button', { name: 'Reset' }).click();
  await page.waitForSelector('.race-view');
  await page.getByRole('button', { name: 'Race' }).click();
  await page.waitForFunction(() => document.querySelector('.legend')?.textContent?.includes('sgd (diverged)'), null, { timeout: 15000 }).catch(() => {});
  check((await page.locator('.legend').textContent())?.includes('sgd (diverged)') ?? false, 'race: sgd at lr 0.11 on the bowl is marked diverged');
  check(errors.length === 0, `race: no console errors${errors.length ? ': ' + errors[0] : ''}`);

  // the overfitting lab: its switch leads to the same network with more data
  await page.goto(`${base}#preset=overfit`);
  await page.waitForSelector('.train-view');
  await page.locator('.variants a', { hasText: '400' }).click();
  await page.waitForFunction(() => location.hash.includes('preset=overfit-more'), null, { timeout: 5000 }).catch(() => {});
  await page.waitForSelector('.train-view h1');
  check(/more data/.test((await page.locator('.train-view h1').textContent()) ?? ''), 'overfit: the switch opens the 400-dot lab');

  // a link to a point in a pass opens there, with the backward order shown
  errors.length = 0;
  await page.goto(`${base}#preset=expression&pass=reverse&i=7`);
  await page.waitForSelector('.react-flow__node');
  await page.waitForFunction(() => document.querySelector('.transport .counter')?.textContent?.startsWith('7 /'), null, { timeout: 10000 }).catch(() => {});
  check((await page.locator('.transport .counter').textContent())?.startsWith('7 /') ?? false, 'link: opens at step 7');
  const queue = await page.locator('.queue li').allTextContents();
  check(queue.join('|') === 'ddone|cready|awaiting 1 of 2|bwaiting', `link: backward order at step 7 (${queue.join(', ')})`);
  check(errors.length === 0, `link: no console errors${errors.length ? ': ' + errors[0] : ''}`);

  // chained train steps keep their updates, and the loss falls
  await page.goto(`${base}#preset=neuron&pass=train`);
  await page.reload();
  await page.waitForSelector('.react-flow__node');
  await page.getByTitle('To the end').click();
  for (let k = 0; k < 3; ++k) {
    await page.getByRole('button', { name: 'Next train step' }).click();
    await page.waitForFunction(() => document.querySelector('.transport .counter')?.textContent?.startsWith('0 /'), null, { timeout: 5000 }).catch(() => {});
    await page.getByTitle('To the end').click();
    await sleep(300);
  }
  const lossText = (await page.locator('.train-steps p').textContent()) ?? '';
  const losses = lossText.split(':')[1]?.split('→').map((x) => Number(x.trim())) ?? [];
  check(losses.length === 3 && losses[2] < losses[0], `train steps: the loss falls over three steps (${losses.join(' → ')})`);
  await page.goto(`${base}#preset=expression&pass=reverse&i=7`);
  await page.reload();
  await page.waitForSelector('.react-flow__node');

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
  if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}
console.log(`=== ${failed ? `${failed} failed` : 'all passed'} ===`);
process.exit(failed ? 1 : 0);
