import { PRESETS } from '../graph/presets.ts';
import type { PresetMode } from '../graph/presets.ts';
import { BASICS, MODE_INTRO } from '../content/help.tsx';

const SECTIONS: { mode: PresetMode; title: string }[] = [
  { mode: 'step', title: 'Step through' },
  { mode: 'train', title: 'Train' },
  { mode: 'race', title: 'Race optimizers' },
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
        <p>New to this? Read the six ideas below, then open <a href="#preset=expression">d = a·b + a</a> and press → a few times.</p>
      </header>
      <section className="basics">
        <h2>The six ideas this site shows</h2>
        <dl>
          {BASICS.map((b) => (
            <div key={b.term}><dt>{b.term}</dt><dd>{b.text}</dd></div>
          ))}
        </dl>
      </section>
      {SECTIONS.map((s) => (
        <section key={s.mode} className="preset-section">
          <h2>{s.title}</h2>
          <p className="muted">{MODE_INTRO[s.mode]}</p>
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
