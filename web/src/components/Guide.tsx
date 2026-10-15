// The explanation card at the top of every example: what it shows and what to
// try. Open on a first visit; once closed it stays closed (remembered in this
// browser only, and the page works the same without that memory).

import { useState } from 'react';
import type { ReactNode } from 'react';
import type { Guide } from '../content/help.tsx';
import { lessonForPreset } from '../content/lessons/index.ts';

const KEY = 'ag.guide.closed';

function readClosed(): boolean {
  try {
    return localStorage.getItem(KEY) === '1';
  } catch {
    return false;
  }
}

export function GuideCard(props: { title: string; presetId: string; guide?: Guide; children?: ReactNode }) {
  const lesson = lessonForPreset(props.presetId);
  const [open, setOpen] = useState(() => !readClosed());
  const toggle = (next: boolean) => {
    setOpen(next);
    try {
      localStorage.setItem(KEY, next ? '0' : '1');
    } catch { /* private mode: just not remembered */ }
  };
  return (
    <details className="guide" open={open} onToggle={(e) => toggle((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>About this example: <b>{props.title}</b></summary>
      <div className="guide-body">
        {lesson && (
          <p className="guide-lesson">New to this? Read the lesson first: <a href={`#lesson=${lesson.id}`}>{lesson.title}</a>.</p>
        )}
        {props.guide && <p>{props.guide.idea}</p>}
        {props.children}
        {props.guide && props.guide.watch.length > 0 && (
          <>
            <div className="guide-try">Try this</div>
            <ul>{props.guide.watch.map((w, i) => <li key={i}>{w}</li>)}</ul>
          </>
        )}
      </div>
    </details>
  );
}

/** One line of help under a heading or control. */
export function Help({ children }: { children: ReactNode }) {
  return <p className="help">{children}</p>;
}
