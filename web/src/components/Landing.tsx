// The home page: a learning path from beginner to advanced, each lesson
// linked to labs, and the labs themselves for anyone who wants to play.

import { useState } from 'react';
import { PRESETS } from '../graph/presets.ts';
import type { PresetMode } from '../graph/presets.ts';
import { MODE_INTRO } from '../content/help.tsx';
import { LESSONS, LEVELS, readDone } from '../content/lessons/index.ts';

const SECTIONS: { mode: PresetMode; title: string }[] = [
  { mode: 'step', title: 'Step through' },
  { mode: 'train', title: 'Train' },
  { mode: 'race', title: 'Race optimizers' },
];

export function Landing() {
  const [done] = useState(readDone);
  const firstUnread = LESSONS.find((l) => !done.has(l.id)) ?? LESSONS[0];
  return (
    <div className="landing">
      <header className="hero">
        <h1>Learn how neural networks learn</h1>
        <p>
          A course from first principles to the ideas behind modern training, where every idea comes with something
          to see: every number a network computes and every gradient it sends back, drawn as it happens, on a small
          engine running in your browser.
        </p>
        <p>
          Each lesson starts from a question, tells what people did before and what problem the idea solved, explains how
          and why it works and where it is used today, then sends you to a lab to watch it happen.
        </p>
        <a className="button primary" href={`#lesson=${firstUnread.id}`}>
          {done.size === 0 ? 'Start with lesson 1' : `Continue: ${firstUnread.title}`} →
        </a>
      </header>

      {LEVELS.map((lv) => {
        const lessons = LESSONS.filter((l) => l.level === lv.level);
        return (
          <section key={lv.level} className={`level level-${lv.level}`}>
            <h2>{lv.title}</h2>
            <p className="muted">{lv.line}</p>
            <ol className="lesson-list">
              {lessons.map((l) => (
                <li key={l.id}>
                  <a href={`#lesson=${l.id}`}>
                    <span className="lesson-num">{LESSONS.indexOf(l) + 1}</span>
                    <span>
                      <b>{l.title}</b>{done.has(l.id) && <span className="read" title="read"> ✓</span>}
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
      <p className="muted">Every interactive example on its own, for exploring freely. Each lesson links to the right one.</p>
      {SECTIONS.map((s) => (
        <section key={s.mode} className="preset-section">
          <h3>{s.title}</h3>
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
