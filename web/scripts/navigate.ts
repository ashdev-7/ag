// Clicks through the site the way a person does, without ever reloading:
// every lesson from the home page, every lab each lesson links to, back, the
// preset menu through every lab, the home page's lab cards, variant switches,
// and fast repeated clicks. After each click the page must work: at the top,
// its content drawn, and a lab must actually run.
//
//   npm run build && node scripts/navigate.ts [chromium|firefox|webkit] [base url]
//   SLOW=4 node scripts/navigate.ts chromium     (a slow machine)
//   PROFILE=nowasm node scripts/navigate.ts ...  (a browser missing something: see PROFILES)

import { spawn, spawnSync } from 'node:child_process';
import { chromium, firefox, webkit } from 'playwright';
import type { Page } from 'playwright';
import { PRESETS } from '../src/graph/presets.ts';

// PROFILE: a browser missing something the engine would like. The live
// labs must work in every one, on the path named here:
//   nowasm        no WebAssembly (iOS Lockdown Mode): worker, JavaScript build
//   wasmblocked   WebAssembly present but refused (strict security policy): worker, JavaScript build
//   brokenworker  the worker fails as it starts (Firefox before 114, without module workers): page, WebAssembly
//   noworker      no workers at all: page, WebAssembly
//   bare          no workers and no WebAssembly: page, JavaScript build
//   oldjs         no Array/TypedArray .at() (iOS 15.0-15.3, Chrome before 92): worker, WebAssembly
const PROFILES: Record<string, { expect: string; script: string }> = {
  nowasm: { expect: 'worker js', script: 'NO_WASM' },
  wasmblocked: { expect: 'worker js', script: 'BLOCK_WASM' },
  brokenworker: { expect: 'page wasm', script: 'BROKEN_WORKER' },
  noworker: { expect: 'page wasm', script: 'NO_WORKER' },
  bare: { expect: 'page js', script: 'NO_WORKER NO_WASM' },
  oldjs: { expect: 'worker wasm', script: 'NO_AT' },
};
const profile = process.env.PROFILE ? PROFILES[process.env.PROFILE] : undefined;
if (process.env.PROFILE && !profile) throw new Error(`no profile ${process.env.PROFILE}`);

/** Runs before the page's own scripts: takes away what the profile says, in the page and in its workers. */
function cripple(flags: string) {
  const blockWasm = 'const no = () => { throw new WebAssembly.CompileError("blocked by policy"); };'
    + 'WebAssembly.Module = function () { no(); }; WebAssembly.compile = WebAssembly.instantiate = WebAssembly.compileStreaming'
    + ' = WebAssembly.instantiateStreaming = () => Promise.reject(new WebAssembly.CompileError("blocked by policy"));';
  const inWorker = flags.includes('NO_WASM') ? 'delete self.WebAssembly;' : flags.includes('BLOCK_WASM') ? blockWasm : '';
  const noAt = 'for (const C of [Array, String, Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array]) delete C.prototype.at;';
  const w = window as unknown as Record<string, unknown>;
  if (flags.includes('NO_AT')) new Function(noAt)();
  if (flags.includes('NO_WASM')) delete w.WebAssembly;
  if (flags.includes('BLOCK_WASM')) new Function(blockWasm)();
  const Real = window.Worker;
  if (flags.includes('NO_WORKER')) { delete w.Worker; return; }
  w.Worker = function (url: string | URL, opts?: WorkerOptions) {
    // a browser without module workers runs the script as a classic one, which stops at its first import
    if (flags.includes('BROKEN_WORKER'))
      return new Real(URL.createObjectURL(new Blob(['throw new SyntaxError("import declarations may only appear at top level of a module")'], { type: 'text/javascript' })));
    const href = new URL(url, location.href).href;
    // before the worker's own code: a module of its own, imported first
    if (flags.includes('NO_AT')) {
      const first = URL.createObjectURL(new Blob([noAt], { type: 'text/javascript' }));
      return new Real(URL.createObjectURL(new Blob([`import ${JSON.stringify(first)};
import ${JSON.stringify(href)};`], { type: 'text/javascript' })), opts);
    }
    if (!inWorker) return new Real(url, opts);
    // a static import runs first, then this script; the engine loads later, on 'init'
    const src = `import ${JSON.stringify(href)};
${inWorker}`;
    return new Real(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })), opts);
  };
}

const browserName = process.argv[2] ?? 'chromium';
const browserType = { chromium, firefox, webkit }[browserName]!;
const remote = process.argv[3];
const PORT = 4175;
const base = remote ?? `http://localhost:${PORT}/`;
const win = process.platform === 'win32';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const server = remote ? null
  : spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });

let failed = 0, passed = 0;
const fail = (what: string) => { failed++; console.log(`  FAIL  ${what}`); };
let page: Page;
const errors: string[] = [];

/** Waits for `ok` to hold, up to `ms`. */
async function until(ok: () => Promise<boolean>, ms = 8000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if (await ok()) return true; } catch { /* the page is changing under us */ }
    await sleep(100);
  }
  return false;
}

const atTop = () => page.evaluate(() => window.scrollY < 40);
const count = (sel: string) => page.locator(sel).count();
const stepOf = async (sel: string) => Number(/(\d+)/.exec((await page.locator(sel).first().textContent()) ?? '')?.[1] ?? -1);

/** The page now showing works: checked by what kind of page it should be. */
async function works(what: string, hash: string, top = true) {
  const want = hash.split('&')[0];
  const there = () => page.evaluate((w) => w === '' || w === '#' ? window.location.hash === '' || window.location.hash === '#' : window.location.hash.startsWith(w), want);
  if (!(await until(there, 1000))) {
    const h = await page.evaluate(() => window.location.hash);
    return fail(`${what}: address is ${h || '(home)'}, expected ${hash}`);
  }
  if (top && !(await until(atTop, 3000))) fail(`${what}: opened scrolled down, not at the top`);
  const banner = await page.locator('.banner-error').count();
  if (banner) return fail(`${what}: error banner: ${await page.locator('.banner-error').first().textContent()}`);
  const preset = PRESETS.find((p) => hash.startsWith(`#preset=${p.id}`));
  let ok = true;
  if (!hash || hash === '#') ok = await until(async () => (await count('.lesson-list a')) > 0);
  else if (hash.startsWith('#lesson=')) ok = await until(async () => (await count('.lesson h1')) === 1 && (await count('.lesson .labs')) === 1);
  else if (preset?.mode === 'step') {
    ok = await until(async () => (await count('.react-flow__node')) > 0 && (await count('.react-flow__edge')) > 0);
    if (ok) {
      const before = await stepOf('.transport .counter');
      await page.locator('.transport').getByRole('button', { name: 'One step forward' }).click();
      ok = await until(async () => (await stepOf('.transport .counter')) === before + 1, 3000);
      if (!ok) return fail(`${what}: drawn, but a step forward did nothing`);
    }
  } else if (preset?.mode === 'train') {
    ok = await until(async () => (await count('.boundary canvas')) > 0 && (await count('.toolbar .step-count')) === 1);
    if (ok) {
      await page.locator('.toolbar').getByRole('button', { name: 'Step', exact: true }).click();
      ok = await until(async () => (await stepOf('.toolbar .step-count')) >= 1, 4000);
      if (!ok) return fail(`${what}: drawn, but Step did not train`);
    }
  } else if (preset?.mode === 'race') {
    ok = await until(async () => (await count('.race-grid canvas')) > 0 && (await count('.toolbar .step-count')) === 1);
    if (ok) {
      await page.locator('.toolbar').getByRole('button', { name: 'Race' }).click();
      ok = await until(async () => (await stepOf('.toolbar .step-count')) >= 1, 4000);
      await page.locator('.toolbar').getByRole('button', { name: 'Pause' }).click().catch(() => {});
      if (!ok) return fail(`${what}: drawn, but Race did not move`);
    }
  }
  if (ok && preset) {
    if (await count('.recorded')) return fail(`${what}: showed a recording, not the live lab`);
    const where = await page.locator('.app').getAttribute('data-engine-where');
    if (profile && where !== profile.expect) return fail(`${what}: engine ran as "${where}", expected "${profile.expect}"`);
    if (!profile && where !== 'worker wasm') return fail(`${what}: engine ran as "${where}", expected "worker wasm"`);
  }
  if (!ok) {
    const text = (await page.locator('main').innerText().catch(() => '')).slice(0, 80).replace(/\s+/g, ' ');
    return fail(`${what}: did not load (main shows "${text}")`);
  }
  passed++;
}

/** Scrolls a link into view the way a reader would, then clicks it. */
async function clickLink(loc: ReturnType<Page['locator']>) {
  await loc.scrollIntoViewIfNeeded();
  await sleep(150);
  await loc.click();
}

try {
  if (server) for (let i = 0; i < 40; ++i) { try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ } await sleep(250); }
  const browser = await browserType.launch();
  console.log(`browser: ${browserName} ${browser.version()}  site: ${base}`);
  page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  if (profile) {
    await page.addInitScript(cripple, profile.script);
    console.log(`profile ${process.env.PROFILE}: the engine should run in the ${profile.expect.replace(' ', ', ')} build`);
  }
  // SLOW=4: a CPU four times slower (Chromium only), for timing bugs a fast machine hides
  if (process.env.SLOW && browserName === 'chromium') {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: Number(process.env.SLOW) });
    console.log(`CPU slowed ${process.env.SLOW}x`);
  }
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(base);
  await until(async () => (await count('.splash')) === 0, 6000);   // a first visit's opening screen
  await works('home', '');

  // 1. every lesson from the home page, every lab it links to, and back
  const lessons = await page.locator('.lesson-list a').evaluateAll((as) => as.map((a) => a.getAttribute('href')!));
  console.log(`lessons and their labs (${lessons.length} lessons)`);
  for (const href of lessons) {
    await clickLink(page.locator('nav.top .brand'));
    await works(`home (before ${href})`, '');
    await clickLink(page.locator(`.lesson-list a[href="${href}"]`));
    await works(`lesson ${href}`, href);
    const labs = await page.locator('.lesson .labs a.button').evaluateAll((as) => as.map((a) => a.getAttribute('href')!));
    for (const lab of labs) {
      await clickLink(page.locator(`.lesson .labs a.button[href="${lab}"]`).first());
      await works(`${href} → ${lab}`, lab);
      await page.goBack();
      await works(`back to ${href} from ${lab}`, href, false);   // back returns to where the reader was
    }
  }

  // 2. next-lesson links, start to end
  console.log('next-lesson links');
  await page.locator('nav.top .brand').click();
  await clickLink(page.locator('.hero a.button.primary'));
  await works('Start with lesson 1', lessons[0]);
  for (let i = 1; i < lessons.length; ++i) {
    await clickLink(page.locator('.lesson-nav a').last());
    await works(`next → ${lessons[i]}`, lessons[i]);
  }

  // 3. every lab card on the home page
  console.log('lab cards on the home page');
  for (const p of PRESETS) {
    await clickLink(page.locator('nav.top .brand'));
    await works('home', '');
    await clickLink(page.locator(`a.card[href="#preset=${p.id}"]`));
    await works(`card → ${p.id}`, `#preset=${p.id}`);
  }

  // 4. the preset menu, through every lab in turn
  console.log('the lab menu');
  for (const p of PRESETS) {
    await page.locator('nav.top select').selectOption(p.id);
    await works(`menu → ${p.id}`, `#preset=${p.id}`);
  }

  // 5. variant switches inside a lab
  console.log('variant switches');
  await page.locator('nav.top select').selectOption('overfit');
  await works('menu → overfit', '#preset=overfit');
  for (const id of ['overfit-more', 'overfit', 'overfit-more']) {
    await clickLink(page.locator(`.variants a[href="#preset=${id}"]`));
    await works(`variant → ${id}`, `#preset=${id}`);
  }

  // 6. impatient clicking: five labs in a row without waiting, then the last must work
  console.log('fast clicking');
  for (let round = 0; round < 3; ++round) {
    const ids = PRESETS.filter((_, i) => (i + round) % 3 === 0).slice(0, 5).map((p) => p.id);
    for (const id of ids) { await page.evaluate((id) => { window.location.hash = `#preset=${id}`; }, id); await sleep(60); }
    await works(`fast clicks ending at ${ids.at(-1)}`, `#preset=${ids.at(-1)}`);
  }
  // back and forward through history
  for (let i = 0; i < 6; ++i) await page.goBack();
  await sleep(300);
  await works('six steps back', await page.evaluate(() => window.location.hash), false);
  for (let i = 0; i < 3; ++i) await page.goForward();
  await sleep(300);
  await works('three steps forward', await page.evaluate(() => window.location.hash), false);

  const real = errors.filter((e) => !/favicon/.test(e));
  if (real.length) { failed++; console.log(`  FAIL  console errors: ${[...new Set(real)].slice(0, 5).join(' | ')}`); }
  await browser.close();
} finally {
  if (server) {
    if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
    else process.kill(-server.pid!, 'SIGTERM');
  }
}
console.log(`=== ${passed} pages worked, ${failed} failed ===`);
process.exit(failed ? 1 : 0);
