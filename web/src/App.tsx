// The shell: one engine client for the page, the preset and seed from the
// URL, one error banner, and the view for the preset's mode. When the engine
// restarts, the view is remounted and rebuilds its sessions from the spec and
// seed, so a crash costs the current run and nothing else.

import { useCallback, useEffect, useMemo, useState } from 'react';
import { createEngineClient } from './engine/client.ts';
import type { EngineStatus } from './engine/client.ts';
import { findPreset, PRESETS } from './graph/presets.ts';
import { fromUrlHash, toUrlHash } from './views/graph.ts';
import type { UrlState } from './views/graph.ts';
import { describe, ErrorBanner } from './components/ErrorBanner.tsx';
import type { Banner } from './components/ErrorBanner.tsx';
import { Landing } from './components/Landing.tsx';
import { RaceView } from './components/RaceView.tsx';
import { Recorded } from './components/Recorded.tsx';
import { StepView } from './components/StepView.tsx';
import { TrainView } from './components/TrainView.tsx';

const MODE_LABEL = { step: 'Step through', train: 'Train', race: 'Race' };

export function App() {
  const [status, setStatus] = useState<EngineStatus>({ state: 'loading' });
  const client = useMemo(() => createEngineClient({
    createWorker: () => new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' }),
    wasmUrl: new URL(`${import.meta.env.BASE_URL}engine/ag.mjs`, window.location.href).href,
    onStatus: setStatus,
  }), []);
  useEffect(() => () => client.dispose(), [client]);

  const [url, setUrl] = useState<UrlState>(() => fromUrlHash(window.location.hash));
  useEffect(() => {
    const onHash = () => setUrl(fromUrlHash(window.location.hash));
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  // views report their shareable state; written without adding history entries
  const updateUrl = useCallback((u: Partial<UrlState>) => {
    setUrl((cur) => {
      const next = { ...cur, ...u };
      history.replaceState(null, '', toUrlHash(next));
      return next;
    });
  }, []);

  const [banner, setBanner] = useState<Banner | undefined>();
  const onBanner = useCallback((b?: Banner) => setBanner(b), []);
  useEffect(() => {
    if (status.state === 'restarting') setBanner({ kind: 'error', text: describe(status.error), error: status.error });
    if (status.state === 'unavailable') setBanner({ kind: 'error', text: describe({ ...status.error, code: 'WasmLoadFailed' }), error: status.error });
  }, [status]);

  const preset = findPreset(url.preset);
  const seed = url.seed ?? preset?.config.seed ?? 1;

  const pick = (id: string) => {
    setBanner(undefined);
    window.location.hash = toUrlHash({ preset: id });
  };

  let body;
  if (!preset) body = <Landing />;
  else if (status.state === 'unavailable') body = <Recorded preset={preset} />;
  else if (status.state !== 'ready') body = <div className="loading">Starting the engine…</div>;
  else {
    const key = `${preset.id}:${seed}:${status.generation}`;
    const common = { client, preset, seed, url, onUrl: updateUrl, onBanner };
    body = preset.mode === 'step' ? <StepView key={key} {...common} />
      : preset.mode === 'train' ? <TrainView key={key} {...common} />
      : <RaceView key={key} client={client} preset={preset} seed={seed} onBanner={onBanner} />;
  }

  return (
    <div className="app">
      <nav className="top">
        <a className="brand" href="#">ag<span>·glass</span></a>
        {preset && (
          <>
            <select value={preset.id} onChange={(e) => pick(e.target.value)} aria-label="Preset">
              {(['step', 'train', 'race'] as const).map((m) => (
                <optgroup key={m} label={MODE_LABEL[m]}>
                  {PRESETS.filter((p) => p.mode === m).map((p) => <option key={p.id} value={p.id}>{p.title}</option>)}
                </optgroup>
              ))}
            </select>
            <span className="blurb">{preset.blurb}</span>
            {preset.mode !== 'race' && (
              <label className="field">seed
                <input type="number" value={seed} min={0} onChange={(e) => updateUrl({ seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })} />
              </label>
            )}
          </>
        )}
        <span className={`engine-dot engine-${status.state}`} title={`engine: ${status.state}`} />
      </nav>
      <ErrorBanner banner={banner} onDismiss={() => setBanner(undefined)} />
      <main>{body}</main>
    </div>
  );
}
