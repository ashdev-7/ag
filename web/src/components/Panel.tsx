// One panel of a lab: a number, a short title, one line saying how to read
// it, then the picture. Every panel has the same shape, so a reader learns
// the layout once.

import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';

export function Panel(props: {
  n?: string; title: ReactNode; sub?: ReactNode; aside?: ReactNode; className?: string; children: ReactNode;
}) {
  return (
    <section className={`panel ${props.className ?? ''}`}>
      <header className="panel-head">
        {props.n && <span className="pn" aria-hidden>{props.n}</span>}
        <div>
          <h3>{props.title}</h3>
          {props.sub && <p className="sub">{props.sub}</p>}
        </div>
        {props.aside && <div className="aside">{props.aside}</div>}
      </header>
      {props.children}
    </section>
  );
}

/** What a chart shows right now, in words, updated as it changes. */
export function Story({ children, top }: { children: ReactNode; top?: boolean }) {
  // the words change as the numbers do, and a line more or less would move
  // everything below. The box only grows, so the page never jumps back and
  // forth under the reader; a new width starts it over
  // (watched, not measured on each change: measuring would force a layout)
  const box = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    const p = box.current;
    if (!p) return;
    let tallest = 0, width = 0;
    const ro = new ResizeObserver(([e]) => {
      const r = e.target.getBoundingClientRect();
      if (Math.abs(r.width - width) > 1) { width = r.width; tallest = 0; p.style.minHeight = ''; return; }
      if (r.height > tallest + 0.5) { tallest = r.height; p.style.minHeight = `${tallest}px`; }
    });
    ro.observe(p);
    return () => ro.disconnect();
  }, []);
  // not aria-live: during training it changes several times a second
  return <p ref={box} className={`story${top ? ' top' : ''}`}>{children}</p>;
}
