import { PRESETS } from '../graph/presets.ts';
import type { PresetMode } from '../graph/presets.ts';

const SECTIONS: { mode: PresetMode; title: string; line: string }[] = [
  { mode: 'step', title: 'Step through', line: 'One micro-step at a time: every value forward, every gradient back, edge by edge.' },
  { mode: 'train', title: 'Train', line: 'Watch the boundary bend, the loss fall and each layer’s gradients, live.' },
  { mode: 'race', title: 'Race optimizers', line: 'SGD, momentum, RMSProp and Adam from the same start on the same surface.' },
];

export function Landing() {
  return (
    <div className="landing">
      <header className="hero">
        <h1>A glass box for neural networks</h1>
        <p>
          Every number a network computes, every gradient it sends back, every step an optimizer takes, drawn as it
          happens. The engine is a small C++ autodiff library compiled to WebAssembly; it runs here, in your browser.
        </p>
      </header>
      {SECTIONS.map((s) => (
        <section key={s.mode} className="preset-section">
          <h2>{s.title}</h2>
          <p className="muted">{s.line}</p>
          <div className="cards">
            {PRESETS.filter((p) => p.mode === s.mode).map((p) => (
              <a key={p.id} className="card" href={`#preset=${p.id}`}>
                <h3>{p.title}</h3>
                <p>{p.blurb}</p>
              </a>
            ))}
          </div>
        </section>
      ))}
      <footer className="foot muted">
        Built on ideas from micrograd, TensorFlow Playground, CNN Explainer, Transformer Explainer and Netron.
        Graph drawing by React Flow and dagre.
      </footer>
    </div>
  );
}
