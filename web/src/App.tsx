// The shell: one engine client for the page, the preset and seed from the
// URL, one error banner, and the view for the preset's mode. When the engine
// restarts, the view is remounted and rebuilds its sessions from the spec and
// seed, so a crash costs the current run and nothing else.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createEngineClient } from './engine/client.ts';
import { createLocalWorker } from './engine/local.ts';
import type { EngineStatus } from './engine/client.ts';
import { findPreset, PRESETS } from './graph/presets.ts';
import { fromUrlHash, toUrlHash } from './views/graph.ts';
import type { UrlState } from './views/graph.ts';
import { describe, ErrorBanner } from './components/ErrorBanner.tsx';
import type { Banner } from './components/ErrorBanner.tsx';
import { Landing } from './components/Landing.tsx';
import { LessonView } from './components/LessonView.tsx';
import { findLesson } from './content/lessons/index.ts';
import { RaceView } from './components/RaceView.tsx';
import { Recover } from './components/Recover.tsx';
import { Recorded } from './components/Recorded.tsx';
import { StepView } from './components/StepView.tsx';
import { TrainView } from './components/TrainView.tsx';

const MODE_LABEL = { step: 'Step through', train: 'Train', race: 'Race' };

export function App() {
  const [status, setStatus] = useState<EngineStatus>({ state: 'loading' });
  const client = useMemo(() => createEngineClient({
    createWorker: () => new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' }),
    createLocal: createLocalWorker,
    wasmUrl: new URL(`${import.meta.env.BASE_URL}engine/ag.mjs`, window.location.href).href,
    jsUrl: new URL(`${import.meta.env.BASE_URL}engine/ag-js.mjs`, window.location.href).href,
    onStatus: setStatus,
  }), []);
  useEffect(() => () => client.dispose(), [client]);

  const [url, setUrl] = useState<UrlState>(() => fromUrlHash(window.location.hash));
  // a page opened by a click (a link, the menu) starts at the top; back and
  // forward leave the browser to return the reader to where they were
  const clicked = useRef(false);
  // every arrival by address (a link, back, forward, an edited address) opens
  // the page fresh from it, even one for the lab already showing: the views'
  // own writes replace the address and fire no hashchange
  const [visit, setVisit] = useState(0);
  useEffect(() => {
    const onHash = () => {
      setUrl(fromUrlHash(window.location.hash));
      setVisit((v) => v + 1);
    };
    const onClick = (e: MouseEvent) => {
      const a = (e.target as Element | null)?.closest?.('a[href^="#"]');
      if (a && !e.defaultPrevented && e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey) clicked.current = true;
    };
    window.addEventListener('hashchange', onHash);
    document.addEventListener('click', onClick, true);
    return () => {
      window.removeEventListener('hashchange', onHash);
      document.removeEventListener('click', onClick, true);
    };
  }, []);
  const page = url.lesson ? `lesson:${url.lesson}` : url.preset ? `preset:${url.preset}` : 'home';
  useLayoutEffect(() => {
    if (clicked.current) window.scrollTo(0, 0);
    clicked.current = false;
  }, [page]);

  // views report their shareable state, written without adding history
  // entries. Only the view for the page in the address may write: a view
  // that is being left can still finish a step after the reader has moved
  // on, and its write would put its own address over the page they chose
  const updateUrl = useCallback((owner: string, u: Partial<UrlState>) => {
    const here = fromUrlHash(window.location.hash);
    if (here.preset !== owner || here.lesson) return;
    setUrl((cur) => {
      if (cur.preset !== owner || cur.lesson) return cur;
      const next = { ...cur, ...u };
      history.replaceState(null, '', toUrlHash(next));
      return next;
    });
  }, []);

  const [banner, setBanner] = useState<Banner | undefined>();
  const onBanner = useCallback((b?: Banner) => setBanner(b), []);
  // a crashed engine restarts by itself and the lab rebuilds its run: that
  // is ours to handle, so the reader is not told. Only a browser that cannot
  // run the labs at all gets one plain sentence
  useEffect(() => {
    if (status.state === 'unavailable') setBanner({ kind: 'info', text: describe({ ...status.error, code: 'WasmLoadFailed' }) });
  }, [status]);

  const preset = findPreset(url.preset);
  const seed = url.seed ?? preset?.config.seed ?? 1;
  const presetId = preset?.id ?? '';
  const onUrl = useCallback((u: Partial<UrlState>) => updateUrl(presetId, u), [updateUrl, presetId]);

  const pick = (id: string) => {
    setBanner(undefined);
    clicked.current = true;
    window.location.hash = toUrlHash({ preset: id });
  };

  const lesson = findLesson(url.lesson);
  let body;
  // lessons are text and small figures: they need no engine
  if (lesson) body = <LessonView lesson={lesson} />;
  else if (!preset) body = <Landing />;
  else if (status.state === 'unavailable') body = <Recorded preset={preset} />;
  else if (status.state !== 'ready') body = <div className="loading" role="status" aria-label="Loading"><span /><span /><span /></div>;
  else {
    const key = `${preset.id}:${seed}:${status.generation}:${visit}`;
    const common = { client, preset, seed, url, onUrl, onBanner };
    body = preset.mode === 'step' ? <StepView key={key} {...common} />
      : preset.mode === 'train' ? <TrainView key={key} {...common} />
      : <RaceView key={key} client={client} preset={preset} seed={seed} onBanner={onBanner} />;
  }

  return (
    <div className="app" data-engine={status.state} data-engine-where={status.state === 'ready' ? `${status.where} ${status.build}` : undefined}>
      <nav className="top">
        <a className="brand" href="#">ag<span>·glass</span></a>
        {lesson && <a className="nav-link" href="#">All lessons</a>}
        {preset && (
          <>
            <select value={preset.id} onChange={(e) => pick(e.target.value)} aria-label="Preset">
              {(['step', 'train', 'race'] as const).map((m) => (
                <optgroup key={m} label={MODE_LABEL[m]}>
                  {PRESETS.filter((p) => p.mode === m).map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
                </optgroup>
              ))}
            </select>
            <span className="blurb" />
            {preset.mode !== 'race' && (
              <label className="field">seed
                <input type="number" value={seed} min={0} onChange={(e) => onUrl({ seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })} />
              </label>
            )}
          </>
        )}
      </nav>
      <ErrorBanner banner={banner} onDismiss={() => setBanner(undefined)} />
      {/* a page that fails to draw keeps the rest of the site; a new page starts clean */}
      <main><Recover key={`${page}:${visit}`}>{body}</Recover></main>
    </div>
  );
}
