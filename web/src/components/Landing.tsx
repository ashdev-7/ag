// The home page: a learning path from beginner to advanced, each lesson
// linked to labs, and the labs themselves for anyone who wants to play.

import { PRESETS } from '../graph/presets.ts';
import type { PresetMode } from '../graph/presets.ts';
import { MODE_INTRO } from '../content/help.tsx';
import { LESSONS, LEVELS } from '../content/lessons/index.ts';
import { HeroNet } from './HeroNet.tsx';

const SECTIONS: { mode: PresetMode; title: string }[] = [
  { mode: 'step', title: 'Step through' },
  { mode: 'train', title: 'Train' },
  { mode: 'race', title: 'Race optimizers' },
];

export function Landing() {
  return (
    <div className="landing">
      <header className="hero">
        <div className="hero-text">
          <span className="eyebrow">A course you can watch</span>
          <h1>See <span>inside</span> a neural network</h1>
          <p>
            From first principles to modern training, with every idea drawn live: each number a network computes and every
            gradient it sends back, on a small engine running in your browser.
          </p>
          <div className="hero-actions">
            <a className="button primary" href={`#lesson=${LESSONS[0].id}`}>Start with lesson 1 →</a>
            <a className="button" href="#preset=moons">Watch a network train</a>
          </div>
        </div>
        <HeroNet />
      </header>

      {LEVELS.map((lv) => {
        const lessons = LESSONS.filter((l) => l.level === lv.level);
        return (
          <section key={lv.level} className={`level level-${lv.level}`}>
            <div className="level-head">
              <h2>{lv.title}</h2>
              <p>{lv.line}</p>
            </div>
            <ol className="lesson-list">
              {lessons.map((l) => (
                <li key={l.id}>
                  <a href={`#lesson=${l.id}`}>
                    <span className="lesson-num">{LESSONS.indexOf(l) + 1}</span>
                    <span>
                      <b>{l.title}</b>
                      <span className="muted">{l.summary}</span>
                    </span>
                  </a>
                </li>
              ))}
            </ol>
          </section>
        );
      })}

      <h2 className="labs-title">The labs</h2>
      <p className="labs-lede">Every interactive example on its own, to explore freely.</p>
      {SECTIONS.map((s) => (
        <section key={s.mode} className="preset-section">
          <div>
            <h3>{s.title}</h3>
            <p>{MODE_INTRO[s.mode]}</p>
          </div>
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
