// The engine in a worker, off the page's thread: the usual home for host.ts.

import '../polyfills.ts';
import { startHost } from './host.ts';
import type { HostScope } from './host.ts';

startHost(self as unknown as HostScope);
