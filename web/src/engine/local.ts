// The engine on the page's own thread, for browsers that cannot start the
// worker (no module workers, workers blocked). It looks like a Worker to the
// client: messages go each way on a later task, as a worker's would, so the
// client cannot tell the difference. Slower to draw while training, but live.

import { startHost } from './host.ts';
import type { HostScope } from './host.ts';
import type { Envelope, FromWorker } from './protocol.ts';

export function createLocalWorker(): Worker {
  let closed = false;
  const page = {
    onmessage: null as ((ev: { data: FromWorker }) => void) | null,
    onerror: null as unknown,
    postMessage(data: Envelope) {
      if (!closed) setTimeout(() => !closed && host.onmessage?.({ data }), 0);
    },
    terminate() {
      closed = true;
    },
  };
  const host: HostScope = {
    onmessage: null,
    postMessage(data: FromWorker) {
      if (!closed) setTimeout(() => !closed && page.onmessage?.({ data }), 0);
    },
    // the engine trapped and has said so in a 'fatal' message: a worker
    // would end here, so this one stops answering too
    close() {
      setTimeout(() => (closed = true), 0);
    },
  };
  startHost(host);
  return page as unknown as Worker;
}
