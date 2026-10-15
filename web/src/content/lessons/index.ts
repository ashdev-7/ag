import { ADVANCED } from './advanced.tsx';
import { BEGINNER } from './beginner.tsx';
import { INTERMEDIATE } from './intermediate.tsx';
import type { Lesson, Level } from './types.ts';

export type { Lab, Lesson, Level } from './types.ts';

/** The whole course, in reading order. */
export const LESSONS: Lesson[] = [...BEGINNER, ...INTERMEDIATE, ...ADVANCED];

export const LEVELS: { level: Level; title: string; line: string }[] = [
  { level: 'beginner', title: 'Beginner', line: 'What a network is, up to backpropagation. School algebra is enough.' },
  { level: 'intermediate', title: 'Intermediate', line: 'How real networks are built and trained.' },
  { level: 'advanced', title: 'Advanced', line: 'Why training goes wrong, and the ideas that fixed it.' },
];

export function findLesson(id: string | undefined): Lesson | undefined {
  return LESSONS.find((l) => l.id === id);
}

/** The lesson that introduces a preset: the first whose labs use it. */
export function lessonForPreset(presetId: string): Lesson | undefined {
  return LESSONS.find((l) => l.labs.some((lab) => lab.preset === presetId));
}

export function labHref(preset: string, hash?: string): string {
  return `#preset=${preset}${hash ? '&' + hash : ''}`;
}

// lessons marked as read: remembered in this browser only, as a convenience
const DONE_KEY = 'ag.lessons.done';

export function readDone(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(DONE_KEY) ?? '[]') as string[]);
  } catch {
    return new Set();
  }
}

export function markDone(id: string, done: boolean): Set<string> {
  const s = readDone();
  if (done) s.add(id); else s.delete(id);
  try {
    localStorage.setItem(DONE_KEY, JSON.stringify([...s]));
  } catch { /* not remembered */ }
  return s;
}
