// The few newer built-ins the site uses, for the browsers that came just
// before them (iOS 15.0-15.3, Chrome before 92). Loaded first by the page and
// by the engine host, so nothing else has to think about it.

function at<T>(this: ArrayLike<T>, i: number): T | undefined {
  const n = this.length;
  const k = Math.trunc(i) || 0;
  const j = k < 0 ? n + k : k;
  return j < 0 || j >= n ? undefined : this[j];
}

for (const C of [Array, String, Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array]) {
  const proto = C.prototype as unknown as { at?: unknown };
  if (typeof proto.at !== 'function')
    Object.defineProperty(proto, 'at', { value: at, writable: true, configurable: true });
}

export {};
