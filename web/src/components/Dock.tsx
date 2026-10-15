// A lab's main buttons, pinned to the bottom of a phone's screen, so a run
// can be started, paused and stepped while its charts are in view. Hidden on
// wider screens, where the controls sit beside what they control. The lab's
// own control row hides these same buttons on a phone (styles.css).

import type { ReactNode } from 'react';

export function Dock({ children }: { children: ReactNode }) {
  return (
    <>
      <div className="dock-space" />
      <div className="dock" role="toolbar" aria-label="Lab controls">{children}</div>
    </>
  );
}
