// A CSS media query as React state, so a component can lay itself out
// differently on a phone (the same breakpoint as styles.css).

import { useEffect, useState } from 'react';

export const PHONE = '(max-width: 760px)';
export const TOUCH = '(pointer: coarse)';

export function useMedia(query: string): boolean {
  const get = () => typeof window !== 'undefined' && !!window.matchMedia?.(query).matches;
  const [on, setOn] = useState(get);
  useEffect(() => {
    const m = window.matchMedia?.(query);
    if (!m) return;
    const update = () => setOn(m.matches);
    update();
    m.addEventListener('change', update);
    return () => m.removeEventListener('change', update);
  }, [query]);
  return on;
}
