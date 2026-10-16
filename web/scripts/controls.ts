// Uses every control on every page, one at a time, and checks after each that
// nothing broke: every button, every option of every menu, every slider at
// its ends and middle, awkward values in every number box (0, negative, huge,
// tiny, empty), every checkbox both ways, every box in a graph, and a sweep
// of the pointer over each picture. After each control the lab must still do
// its job: a step forward steps, Step trains, Race moves.
//
//   npm run build && node scripts/controls.ts [chromium|firefox|webkit] [phone]

import { spawn, spawnSync } from 'node:child_process';
import { chromium, devices, firefox, webkit } from 'playwright';
import type { Locator, Page } from 'playwright';
import { PRESETS } from '../src/graph/presets.ts';
import type { Preset } from '../src/graph/presets.ts';

const browserName = process.argv[2] ?? 'chromium';
const phone = process.argv[3] === 'phone';
const browserType = { chromium, firefox, webkit }[browserName]!;
const PORT = 4176;
const base = `http://localhost:${PORT}/`;
const win = process.platform === 'win32';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });

let failed = 0, used = 0;
const fail = (what: string) => { failed++; console.log(`  FAIL  ${what}`); };
let page: Page;
let errors: string[] = [];

// a value that makes training blow up is the reader's choice, and the lab
// says so: that message is expected, not a breakage
const EXPECTED = /became nan or inf|step was undone|diverged/i;

async function until(ok: () => Promise<boolean>, ms = 5000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if (await ok()) return true; } catch { /* the page is changing */ }
    await sleep(80);
  }
  return false;
}
const num = async (loc: Locator) => Number(/(\d+)/.exec((await loc.first().textContent()) ?? '')?.[1] ?? -1);

/** Nothing broke: no script error, no unexpected error banner, the page not blank. */
async function healthy(what: string) {
  await sleep(120);
  const errs = errors.filter((e) => !/favicon/.test(e));
  errors = [];
  if (errs.length) return fail(`${what}: script error: ${errs[0].slice(0, 160)}`), false;
  const banner = page.locator('.banner-error');
  if (await banner.count()) {
    const text = (await banner.first().textContent()) ?? '';
    if (!EXPECTED.test(text)) return fail(`${what}: error banner: ${text.slice(0, 160)}`), false;
    await page.getByRole('button', { name: 'Dismiss' }).click().catch(() => {});
  }
  // a lab rebuilding (a new seed) shows its loader for a moment: give it 5 s
  const filled = await until(async () => (await page.locator('main').innerText().catch(() => '')).trim().length >= 40);
  if (!filled) return fail(`${what}: the page went blank`), false;
  return true;
}

const scope = () => (phone ? page.locator('.dock') : page.locator('.toolbar, .transport').first());

/** The lab still does its job. */
async function alive(p: Preset, what: string) {
  // still on this lab's page, with its buttons? if not, say where the reader ended up
  const own = page.locator(p.mode === 'step' ? (phone ? '.dock' : '.transport') : (phone ? '.dock' : '.toolbar'));
  if (!(await until(async () => (await own.count()) > 0, 5000))) {
    await page.screenshot({ path: `controls-lost-${p.id}.png` }).catch(() => {});
    const text = (await page.locator('main').innerText().catch(() => '')).replace(/\s+/g, ' ').slice(0, 160);
    return void fail(`${what}: the lab's buttons are gone; address ${page.url().split('#')[1] ?? ''}; page says "${text}"`);
  }
  if (p.mode === 'step') {
    const c = page.locator(phone ? '.dock .counter' : '.transport .counter');
    const back = page.getByRole('button', { name: 'One step back' }).last();
    const fwd = page.getByRole('button', { name: 'One step forward' }).last();
    // a lab left playing moves by itself: pause it first, as a reader would before stepping
    const playing = page.locator(phone ? '.dock' : '.transport').getByRole('button', { name: 'Pause', exact: true });
    if (await playing.count()) { await playing.first().click().catch(() => {}); await sleep(300); }
    // the steps can take a moment to arrive: one of the two must come alive
    if (!(await until(async () => (await fwd.isEnabled()) || (await back.isEnabled())))) return void fail(`${what}: neither step button ever worked`);
    if (await fwd.isDisabled()) {   // at the end: step back first, and let it land
      const end = await num(c);
      await back.click();
      if (!(await until(async () => (await num(c)) === end - 1))) return void fail(`${what}: afterwards, a step back did nothing (counter stayed ${await num(c)})`);
    }
    const before = await num(c);
    await fwd.click();
    if (!(await until(async () => (await num(c)) === before + 1))) {
      await page.screenshot({ path: `controls-stuck-${p.id}.png` }).catch(() => {});
      const bar = (await c.first().innerText().catch(() => '?')).replace(/\s+/g, ' ');
      fail(`${what}: afterwards, a step forward did nothing (counter "${bar}", was ${before}; forward ${await fwd.isEnabled() ? 'enabled' : 'disabled'}; address ${page.url().split('#')[1]})`);
    }
  } else {
    const c = page.locator(phone ? '.dock .counter' : '.toolbar .step-count');
    const pause = scope().getByRole('button', { name: 'Pause' });
    if (await pause.count()) { await pause.click(); await sleep(200); }
    if (p.mode === 'race') {
      await scope().getByRole('button', { name: 'Reset' }).click();
      await until(async () => (await num(c)) === 0, 3000);
      await scope().getByRole('button', { name: 'Race' }).click();
      const moved = await until(async () => (await num(c)) > 0);
      await scope().getByRole('button', { name: 'Pause' }).click().catch(() => {});
      if (!moved) fail(`${what}: afterwards, Race did not move`);
    } else {
      const before = await num(c);
      await scope().getByRole('button', { name: 'Step', exact: true }).click();
      if (!(await until(async () => (await num(c)) > before))) {
        // a reader-chosen blow-up stops training until Reset: that is the lab working
        const reset = scope().getByRole('button', { name: 'Reset' });
        if (!(await reset.count())) {
          await page.screenshot({ path: `controls-noreset-${p.id}.png` });
          const bar = await scope().innerText().catch(() => '?');
          return void fail(`${what}: Step did not train and there is no Reset; the bar says "${bar.replace(/\s+/g, ' ').slice(0, 160)}"`);
        }
        await reset.click();
        await sleep(300);
        const b2 = await num(c);
        await scope().getByRole('button', { name: 'Step', exact: true }).click();
        if (!(await until(async () => (await num(c)) > b2))) {
          await page.screenshot({ path: `controls-nostep-${p.id}.png`, fullPage: true }).catch(() => {});
          const step = scope().getByRole('button', { name: 'Step', exact: true });
          const banner = await page.locator('.banner').allInnerTexts().catch(() => []);
          fail(`${what}: afterwards, Step did not train (even after Reset): counter "${await c.first().innerText()}", Step ${await step.isEnabled() ? 'enabled' : 'disabled'}, banner "${banner.join(' / ').slice(0, 120)}", address ${page.url().split('#')[1]}`);
        }
      }
    }
  }
  await healthy(`${what} (checking it still works)`);
}

/** Every control inside main, once each, re-found after every use because the page changes. */
async function useEverything(p: Preset | null, label: string) {
  const done = new Set<string>();
  const key = (l: Locator) => l.evaluate((e) => `${e.tagName}|${e.getAttribute('aria-label') ?? ''}|${(e as HTMLInputElement).type ?? ''}|${e.textContent?.trim().slice(0, 30) ?? ''}|${[...document.querySelectorAll(e.tagName)].indexOf(e)}`);
  for (let pass = 0; pass < 3; ++pass) {   // controls that appear after others (compare, modes) get their turn
    // the lab's controls, and the seed box in the bar above it
    const all = page.locator('main button:visible, main select:visible, main input:visible, nav.top input:visible');
    const n = await all.count();
    let fresh = 0;
    for (let i = 0; i < n; ++i) {
      const el = all.nth(i);
      let k: string;
      try { k = await key(el); } catch { continue; }
      if (done.has(k)) continue;
      done.add(k);
      fresh++;
      const tag = await el.evaluate((e) => e.tagName.toLowerCase()).catch(() => '');
      const type = await el.getAttribute('type').catch(() => null);
      const name = (await el.getAttribute('aria-label').catch(() => null)) ?? (await el.textContent().catch(() => ''))?.trim().slice(0, 24) ?? '';
      const what = `${label}: ${tag}${type ? `[${type}]` : ''} "${name}"`;
      try {
        if (!(await el.isEnabled())) continue;
        // a page busy training can take a moment to answer; 5 s is still a hang
        await el.scrollIntoViewIfNeeded({ timeout: 5000 });
        if (tag === 'button') {
          await el.click({ timeout: 5000 });
          used++; await healthy(what);
        } else if (tag === 'select') {
          const values = await el.locator('option').evaluateAll((os) => os.map((o) => (o as HTMLOptionElement).value));
          for (const v of values) {
            await el.selectOption(v, { timeout: 5000 });
            used++; if (!(await healthy(`${what} = "${v}"`))) break;
          }
        } else if (type === 'range') {
          const [min, max, step] = await el.evaluate((e) => [Number((e as HTMLInputElement).min || 0), Number((e as HTMLInputElement).max || 100), Number((e as HTMLInputElement).step) || 1]);
          // the middle, on the slider's own steps
          for (const v of [max, min, min + Math.round((max - min) / 2 / step) * step]) {
            await el.fill(String(v), { timeout: 5000 });
            used++; if (!(await healthy(`${what} = ${v}`))) break;
          }
        } else if (type === 'number') {
          for (const v of ['0', '-1', '1000000', '0.0000001', '0.5', '', '3']) {
            await el.fill(v, { timeout: 5000 });
            await el.blur();
            used++; if (!(await healthy(`${what} = "${v}"`))) break;
          }
        } else if (type === 'checkbox') {
          await el.click({ timeout: 5000 }); used++; await healthy(`${what} on`);
          await el.click({ timeout: 5000 }); used++; await healthy(`${what} off`);
        }
      } catch (e) {
        fail(`${what}: could not be used: ${String(e).split('\n')[0].slice(0, 140)}`);
      }
    }
    if (p) await alive(p, `${label} after pass ${pass + 1}`);
    if (!fresh) break;
  }
}

/** Every box in a graph clicked, and the pointer swept over every picture. */
async function pointAtEverything(label: string) {
  const nodes = page.locator('.react-flow__node');
  const n = Math.min(await nodes.count(), 40);
  for (let i = 0; i < n; ++i) {
    try {
      await nodes.nth(i).scrollIntoViewIfNeeded({ timeout: 2000 });
      await nodes.nth(i).click({ timeout: 2000, force: true });
      used++;
    } catch { /* a box hidden behind another at this zoom */ }
  }
  if (n) await healthy(`${label}: clicking every graph box`);
  const canvases = page.locator('main canvas:visible');
  const c = await canvases.count();
  for (let i = 0; i < c; ++i) {
    const box = await canvases.nth(i).boundingBox().catch(() => null);
    if (!box) continue;
    await canvases.nth(i).scrollIntoViewIfNeeded().catch(() => {});
    const b2 = (await canvases.nth(i).boundingBox())!;
    for (let k = 1; k < 8; ++k) await page.mouse.move(b2.x + (b2.width * k) / 8, b2.y + (b2.height * ((k * 3) % 8)) / 8);
    if (phone) await page.mouse.click(b2.x + b2.width / 2, b2.y + b2.height / 2);
    used++;
  }
  if (c) await healthy(`${label}: pointer over every picture`);
}

try {
  for (let i = 0; i < 40; ++i) { try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ } await sleep(250); }
  const browser = await browserType.launch();
  const ctx = await browser.newContext(phone ? { ...devices[browserName === 'webkit' ? 'iPhone 13' : 'Pixel 7'] } : { viewport: { width: 1440, height: 900 } });
  await ctx.addInitScript(() => { try { sessionStorage.setItem('ag-splash-seen', '1'); } catch { /* fine */ } });
  page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  console.log(`browser: ${browserName} ${browser.version()}${phone ? ' (phone)' : ''}`);

  // PART=a-b runs labs a to b-1 (and no lessons); PART=lessons runs only the lessons
  const part = process.env.PART ?? '';
  const [from, to] = /^\d+-\d+$/.test(part) ? part.split('-').map(Number) : [0, part === 'lessons' ? 0 : PRESETS.length];
  for (const p of PRESETS.slice(from, to)) {
    await page.goto(`${base}#preset=${p.id}`);
    const ready = await until(async () => (await page.locator('.app[data-engine="ready"]').count()) === 1
      && (await page.locator(p.mode === 'step' ? '.react-flow__node' : 'main canvas').count()) > 0, 15000);
    if (!ready) { fail(`${p.id}: did not load`); continue; }
    const before = used;
    await alive(p, `${p.id} (fresh)`);
    await pointAtEverything(p.id);
    await useEverything(p, p.id);
    await pointAtEverything(`${p.id} (after)`);
    await alive(p, `${p.id} (at the end)`);
    // a long visit must not pile up memory: the page's heap, where the browser reports it
    const heap = await page.evaluate(() => (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize ?? 0);
    console.log(`  ${p.id.padEnd(13)} ${used - before} controls used${heap ? `, page heap ${(heap / 2 ** 20).toFixed(0)} MB` : ''}`);
  }

  // the lessons' own figures: sliders and buttons
  await page.goto(base);
  const lessons = /^\d+-\d+$/.test(part) ? [] : await page.locator('.lesson-list a').evaluateAll((as) => as.map((a) => a.getAttribute('href')!));
  for (const href of lessons) {
    await page.goto(`${base}${href}`);
    await page.waitForSelector('.lesson h1');
    const before = used;
    await useEverything(null, href);
    await pointAtEverything(href);
    await healthy(href);
    if (used > before) console.log(`  ${href.padEnd(28)} ${used - before} controls used`);
  }
  await browser.close();
} finally {
  if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}
console.log(`=== ${used} controls used, ${failed} failed ===`);
process.exit(failed ? 1 : 0);
