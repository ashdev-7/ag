// One panel of a lab: a number, a short title, one line saying how to read
// it, then the picture. Every panel has the same shape, so a reader learns
// the layout once.

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
  // not aria-live: during training it changes several times a second
  return <p className={`story${top ? ' top' : ''}`}>{children}</p>;
}
