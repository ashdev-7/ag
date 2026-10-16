// Every instruction on the site names things a reader must find: a box to
// click (`logits`), a switch (*Single numbers*), a button (Press Run, Press
// ⏭). This opens each lab the way its instructions do and follows them in
// order, as a reader would: each named thing must be there, by that name, at
// that point, and named buttons and switches are pressed before the next
// instruction is checked. Links inside instructions must open a real page.
//
//   npm run build && node scripts/instructions.ts [chromium|firefox|webkit]

import { spawn, spawnSync } from 'node:child_process';
import { chromium, firefox, webkit } from 'playwright';
import type { Page } from 'playwright';

const browserName = process.argv[2] ?? 'chromium';
const browserType = { chromium, firefox, webkit }[browserName]!;
const PORT = 4177;
const base = `http://localhost:${PORT}/`;
const win = process.platform === 'win32';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--port', String(PORT), '--strictPort'], { stdio: 'ignore', detached: !win });

let failed = 0, checked = 0;
const fail = (what: string) => { failed++; console.log(`  FAIL  ${what}`); };

type Kind = 'code' | 'name' | 'press' | 'link';
interface Ref { kind: Kind; text: string }

// a box's name, possibly with one operator (x + b, x·w) or a call (detach(a)); formulas with numbers or ∂ are maths, not names
const BOX = /^([A-Za-z_][\w·']*( [+×·] [A-Za-z_][\w·']*)?|detach\([a-z]\))$/;
// what a reader is told to press or pick
const PRESS = /\b(?:Press(?:es)?|Pick|then) (⏮|◀|▶|⏭|Run|Race|Reset|Step|Play|Pause|Backward|Forward)(?=[\s,.:;)]|$)/g;

/** What each instruction under `selector` names, in reading order. */
async function refsOf(page: Page, selector: string): Promise<{ text: string; refs: Ref[] }[]> {
  const raw = await page.locator(selector).evaluateAll((items) => items.map((li) => {
    const parts: { kind: string; text: string; at: number }[] = [];
    const html = li.innerHTML;
    li.querySelectorAll('a[href^="#"]').forEach((a) => parts.push({ kind: 'link', text: a.getAttribute('href')!, at: html.indexOf(a.outerHTML) }));
    li.querySelectorAll('code').forEach((c) => parts.push({ kind: 'code', text: c.textContent!.trim(), at: html.indexOf(c.outerHTML) }));
    li.querySelectorAll('i, em').forEach((c) => parts.push({ kind: 'name', text: c.textContent!.trim(), at: html.indexOf(c.outerHTML) }));
    return { text: li.textContent!.trim(), html, parts };
  }));
  return raw.map(({ text, html, parts }) => {
    const all = [...parts];
    for (const m of html.matchAll(PRESS)) all.push({ kind: 'press', text: m[1], at: m.index! });
    all.sort((a, b) => a.at - b.at);
    const refs = all.filter((r) => r.kind !== 'code' || BOX.test(r.text)).map((r) => ({ kind: r.kind as Kind, text: r.text }));
    return { text, refs };
  });
}

/** Is `ref` there to be found on the page now showing? */
async function present(page: Page, ref: Ref): Promise<boolean> {
  return page.evaluate((ref) => {
    const visible = (e: Element) => { const r = e.getBoundingClientRect(); const s = getComputedStyle(e); return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && Number(s.opacity) > 0.05; };
    const norm = (s: string | null | undefined) => (s ?? '').replace(/\s+/g, ' ').trim().toLowerCase();
    const want = norm(ref.text);
    const all = [...document.querySelectorAll('main *')];
    const options = [...document.querySelectorAll('main option')].map((o) => norm(o.textContent));
    if (ref.kind === 'code') return all.some((e) => visible(e) && e.children.length === 0 && norm(e.textContent) === want) || options.includes(want);
    if (ref.kind === 'name')
      return all.some((e) => visible(e) && (norm(e.textContent) === want || norm(e.getAttribute('aria-label')) === want || (e.children.length <= 2 && norm(e.textContent).startsWith(want))))
        || options.includes(want);
    return [...document.querySelectorAll('main button, .dock button')].some((b) => visible(b) && (norm(b.textContent) === want || norm(b.getAttribute('aria-label')) === want || norm(b.textContent).startsWith(want)));
  }, ref);
}

/** Does what a reference tells the reader to do: presses the button, picks the switch, ticks the box. */
async function act(page: Page, ref: Ref) {
  if (ref.kind === 'link' || ref.kind === 'code') return;
  const want = ref.text.toLowerCase();
  const buttons = page.locator('main button:visible');
  const n = await buttons.count();
  for (let i = 0; i < n; ++i) {
    const b = buttons.nth(i);
    const t = ((await b.textContent()) ?? '').replace(/\s+/g, ' ').trim().toLowerCase();
    const a = ((await b.getAttribute('aria-label')) ?? '').toLowerCase();
    if (t === want || a === want || t.startsWith(want)) {
      if (!(await b.isEnabled())) return;
      await b.click();
      // a run: let it go for a moment, then pause, as a reader watching would
      if (/^(run|race|play)$/.test(want)) {
        await sleep(1500);
        const pause = page.locator('main button:visible', { hasText: /^\s*Pause/ });
        if (await pause.count()) await pause.first().click();
      }
      await sleep(500);
      return;
    }
  }
  const box = page.locator('main label', { hasText: ref.text }).locator('input[type="checkbox"]');
  if ((await box.count()) && !(await box.first().isChecked())) { await box.first().click(); await sleep(400); }
}

async function checkOn(page: Page, where: string, items: { text: string; refs: Ref[] }[]) {
  for (const it of items) {
    // "choose a layer": the reader picks the first one in the menu
    if (/choose a layer/i.test(it.text)) {
      const menu = page.getByLabel('Layer to inspect');
      if (await menu.count()) { await menu.selectOption({ index: 1 }); await sleep(800); }
      else fail(`${where}: "${it.text.slice(0, 60)}…" says choose a layer, but there is no layer menu`);
    }
    for (const r of it.refs) {
      checked++;
      const short = `"${it.text.slice(0, 90)}${it.text.length > 90 ? '…' : ''}"`;
      if (r.kind === 'link') {
        const p = new URLSearchParams(r.text.slice(1));
        if (!p.get('preset') && !p.get('lesson')) fail(`${where}: ${short} links to ${r.text}, which is not a page`);
        continue;
      }
      // a reader gives the page a moment: up to 3 s for the named thing to appear
      let found = false;
      for (let t = 0; t < 30 && !found; ++t) { found = await present(page, r); if (!found) await sleep(100); }
      if (!found) fail(`${where}: ${short} names ${r.kind === 'press' ? 'Press ' : ''}${r.text}, not on the page at that point`);
      else await act(page, r);
    }
  }
}

const waitLab = (page: Page) => page.waitForFunction(() => document.querySelector('.app')?.getAttribute('data-engine') === 'ready'
  && (document.querySelector('.react-flow__node') || document.querySelector('main canvas')), null, { timeout: 15000 }).then(() => sleep(600));

try {
  for (let i = 0; i < 40; ++i) { try { if ((await fetch(base)).ok) break; } catch { /* not up yet */ } await sleep(250); }
  const browser = await browserType.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.addInitScript(() => { try { sessionStorage.setItem('ag-splash-seen', '1'); } catch { /* fine */ } });
  await page.goto(base);
  const lessons = await page.locator('.lesson-list a').evaluateAll((as) => as.map((a) => a.getAttribute('href')!));
  const presets = await page.locator('a.card').evaluateAll((as) => as.map((a) => a.getAttribute('href')!));

  // each lesson's labs: the steps, followed in order on the lab they open
  for (const href of lessons) {
    await page.goto(base + href);
    await page.waitForSelector('.lesson .labs');
    const labs = await page.locator('.lesson .labs .lab').count();
    for (let k = 0; k < labs; ++k) {
      await page.goto(base + href);
      await page.waitForSelector('.lesson .labs');
      const lab = page.locator('.lesson .labs .lab').nth(k);
      const title = (await lab.locator('h3').textContent())!;
      const items = await refsOf(page, `.lesson .labs .lab:nth-of-type(${k + 1}) li`);
      const link = (await lab.locator('a.button').getAttribute('href'))!;
      await page.goto(base + link);
      await waitLab(page);
      await checkOn(page, `${href} › "${title}" (${link})`, items);
    }
  }
  // each lab's own "Try this", followed on the lab
  for (const href of presets) {
    await page.goto(base + href);
    await waitLab(page);
    await checkOn(page, `${href} › Try this`, await refsOf(page, '.lab-head li, .try li'));
  }
  await browser.close();
} finally {
  if (win) spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else process.kill(-server.pid!, 'SIGTERM');
}
console.log(`=== ${checked} named things checked, ${failed} not found ===`);
process.exit(failed ? 1 : 0);
