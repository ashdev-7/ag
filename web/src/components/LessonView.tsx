// One lesson: the question, the story from the problem to where the idea is
// used, the questions a reader is about to ask, and labs to see it happen.

import { useEffect } from 'react';
import { labHref, LESSONS, LEVELS } from '../content/lessons/index.ts';
import type { Lesson } from '../content/lessons/index.ts';

export function LessonView({ lesson }: { lesson: Lesson }) {
  const i = LESSONS.indexOf(lesson);
  const prev = LESSONS[i - 1], next = LESSONS[i + 1];
  const level = LEVELS.find((l) => l.level === lesson.level)!;
  useEffect(() => window.scrollTo(0, 0), [lesson]);

  return (
    <article className="lesson">
      <nav className="crumbs">
        <a href="#">Learn</a> › {level.title} · lesson {i + 1} of {LESSONS.length}
      </nav>
      <h1>{lesson.title}</h1>
      <p className="lesson-question">{lesson.question}</p>

      {lesson.sections.map((s) => (
        <section key={s.heading}>
          <h2>{s.heading}</h2>
          {s.body}
        </section>
      ))}

      {lesson.faq.length > 0 && (
        <section className="faq">
          <h2>Questions you might have now</h2>
          {lesson.faq.map((f) => (
            <div key={f.q} className="faq-item">
              <h3>{f.q}</h3>
              <p>{f.a}</p>
            </div>
          ))}
        </section>
      )}

      <section className="labs">
        <h2>See it happen</h2>
        {lesson.labs.map((lab) => (
          <div key={lab.label} className="lab">
            <h3>{lab.label}</h3>
            <ol>{lab.steps.map((s, k) => <li key={k}>{s}</li>)}</ol>
            <a className="button primary" href={labHref(lab.preset, lab.hash)}>Open the lab →</a>
          </div>
        ))}
      </section>

      <nav className="lesson-nav">
        {prev ? <a href={`#lesson=${prev.id}`}>← {prev.title}</a> : <span />}
        {next ? <a href={`#lesson=${next.id}`}>{next.title} →</a> : <a href="#">Back to all lessons</a>}
      </nav>
    </article>
  );
}
