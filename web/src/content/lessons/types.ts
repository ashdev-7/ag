// A lesson: one idea, explained from the problem it solves to where it is
// used, with the questions a newcomer is about to ask answered in advance,
// and a lab in the interactive part of the site to see it happen.

import type { ReactNode } from 'react';

export type Level = 'beginner' | 'intermediate' | 'advanced';

export interface Lab {
  /** Preset id, plus any extra URL state (e.g. 'pass=reverse'). */
  preset: string;
  hash?: string;
  label: string;
  /** What to do there, in order. */
  steps: ReactNode[];
}

export interface Lesson {
  id: string;
  level: Level;
  title: string;
  /** One line on the landing page. */
  summary: string;
  /** The question this lesson answers, in the reader's words. */
  question: ReactNode;
  /** Ordered sections: before this, the idea, how it works, why, where it is used. */
  sections: { heading: string; body: ReactNode }[];
  /** Questions a newcomer is likely to ask next, answered in advance. */
  faq: { q: string; a: ReactNode }[];
  labs: Lab[];
}
