import { describe, expect, it } from 'vitest';
import { PRESETS } from '../../graph/presets.ts';
import { fromUrlHash } from '../../views/graph.ts';
import { LESSONS, lessonForPreset } from './index.ts';

describe('the course', () => {
  it('has unique lesson ids, in level order', () => {
    expect(new Set(LESSONS.map((l) => l.id)).size).toBe(LESSONS.length);
    const order = ['beginner', 'intermediate', 'advanced'];
    LESSONS.slice(1).forEach((l, i) => expect(order.indexOf(l.level)).toBeGreaterThanOrEqual(order.indexOf(LESSONS[i].level)));
  });

  it('sends every lab to a preset that exists, with link state the page understands', () => {
    for (const l of LESSONS)
      for (const lab of l.labs) {
        expect(PRESETS.some((p) => p.id === lab.preset), `${l.id} → ${lab.preset}`).toBe(true);
        if (lab.hash) {
          const parsed = fromUrlHash(lab.hash);
          for (const key of new URLSearchParams(lab.hash).keys())
            expect(parsed[key as keyof typeof parsed], `${l.id}: ${key} in ${lab.hash}`).toBeDefined();
        }
      }
  });

  it('introduces every lab in some lesson', () => {
    for (const p of PRESETS) expect(lessonForPreset(p.id), p.id).toBeDefined();
  });

  it('answers questions and offers a lab in every lesson', () => {
    for (const l of LESSONS) {
      expect(l.sections.length, l.id).toBeGreaterThanOrEqual(3);
      expect(l.faq.length, l.id).toBeGreaterThanOrEqual(2);
      expect(l.labs.length, l.id).toBeGreaterThanOrEqual(1);
    }
  });
});
