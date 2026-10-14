import { useEffect, useRef, useState } from 'react';
import { fit } from '../views/draw.ts';

/** A canvas that redraws on every render and on resize. `draw` gets CSS pixel sizes. */
export function Canvas(props: {
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void;
  className?: string;
  height?: number;
  title?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [, setTick] = useState(0);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ro = new ResizeObserver(() => setTick((t) => t + 1));
    ro.observe(c);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const c = ref.current;
    if (!c || c.clientWidth === 0) return;
    const { ctx, w, h } = fit(c);
    props.draw(ctx, w, h);
  });
  return <canvas ref={ref} className={props.className} title={props.title} style={{ width: '100%', height: props.height ?? 160, display: 'block' }} />;
}
