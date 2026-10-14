import type { EngineError } from '../engine/protocol.ts';

export interface Banner {
  kind: 'error' | 'info';
  text: string;
  error?: EngineError;
}

/** The one place errors are shown. Everything that can go wrong ends up here. */
export function ErrorBanner(props: { banner?: Banner; onDismiss: () => void; onRetry?: () => void }) {
  const b = props.banner;
  if (!b) return null;
  return (
    <div className={`banner banner-${b.kind}`} role={b.kind === 'error' ? 'alert' : 'status'}>
      <span>
        {b.text}
        {b.error && b.error.message !== b.text && <span className="banner-detail"> {b.error.message}</span>}
      </span>
      <span className="banner-actions">
        {props.onRetry && <button onClick={props.onRetry}>Retry</button>}
        <button onClick={props.onDismiss} aria-label="Dismiss">×</button>
      </span>
    </div>
  );
}

/** Words for an engine error, for the banner. */
export function describe(e: EngineError): string {
  switch (e.code) {
    case 'NonFinite': return `A value became nan or inf${e.nodeId ? ` at ${e.nodeId}` : ''}. The step was undone.`;
    case 'ShapeMismatch': return `Shapes do not fit${e.nodeId ? ` at ${e.nodeId}` : ''}.`;
    case 'LimitExceeded': return 'Too large for the browser engine.';
    case 'OutOfMemory': return 'The engine is out of memory. Close a comparison or pick a smaller preset.';
    case 'WorkerCrashed':
    case 'Timeout': return 'The engine stopped responding and is restarting. The run starts again from step 0.';
    case 'WasmLoadFailed': return 'The engine could not start in this browser. Showing recorded runs instead.';
    default: return e.message;
  }
}
