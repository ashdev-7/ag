// The speed slider: steps per second, on the ladder in views/speed.ts.

import { SPEEDS, speedIndex } from '../views/speed.ts';

export function SpeedControl(props: { value: number; onChange: (stepsPerSecond: number) => void }) {
  const i = speedIndex(props.value);
  const text = `${SPEEDS[i]} step${SPEEDS[i] === 1 ? '' : 's'} a second`;
  return (
    <label className="field speed" title="How many training steps run each second">speed
      <input type="range" min={0} max={SPEEDS.length - 1} step={1} value={i}
        onChange={(e) => props.onChange(SPEEDS[Number(e.target.value)])}
        aria-label="Training speed" aria-valuetext={text} />
      <span className="speed-value">{text}</span>
    </label>
  );
}
