# ag · glass

A glass box for neural networks. Step through a forward and backward pass one
number at a time, train small networks live, and race optimizers on a loss
surface. Everything runs in the browser on a small C++ autodiff engine
compiled to WebAssembly.

```
engine/     C++20 tensor engine: ops, compiler, executor, optimizers, datasets, C API
web/        the site: Vite, React, React Flow; talks to the engine in a Web Worker
product-dev/  plan, PRD, Phase 1 design, and the running progress log
```

## What Phase 1 does

- **Step through** five presets (an expression, the same with a `detach`, one
  neuron, broadcasting, a two-layer MLP) micro-step by micro-step: every forward value, the seed, each
  gradient contribution along each edge, each parameter update. Scrub back and
  forth; going back is exact.
- **Two zoom levels.** Tensors, or every element as its own node, with the
  chain-rule product written out in numbers.
- **Reverse vs forward mode.** One reverse pass against one forward-mode pass
  per input number, with the derivatives side by side.
- **Train** six presets live: decision boundary, loss, gradient norm per
  layer, activation histograms with the share of dead units, and a weight's
  optimizer state. Two optimizers can train side by side from identical
  weights. Two presets reproduce vanishing gradients and dead ReLUs.
- **Race** SGD, momentum, RMSProp and Adam on three 2-d loss surfaces.
- **Fault tolerance.** A nan or inf stops training at the node where it first
  appeared and the step is rolled back; a crashed or hung engine is restarted;
  if the engine cannot run at all the page shows recorded runs.

## Build and run

Needs a C++20 compiler, `make`, Node 22.18+, and [Emscripten](https://emscripten.org/docs/getting_started/downloads.html)
for the WebAssembly build. Python with PyTorch is only needed for `make torch`.

```sh
make -C engine test       # native tests: every op against finite differences, sessions, faults, C API
make -C engine torch      # every op and optimizer against PyTorch
make -C engine wasm       # engine -> web/public/engine/ag.{mjs,wasm}
make -C engine wasm-test  # the moons check through the WASM build; must agree with native
make -C engine bench      # step time against model size (wasm-bench for the WASM build)

cd web
npm install
npm run dev               # http://localhost:5173
npm test                  # spec validation, URL parsing, engine client recovery
npm run verify            # every step the stepper shows against an independent double-precision reference
npm run behaviour         # every training lab and race does what its text says, measured in the engine
npm run build             # dist/
node scripts/smoke.ts shots firefox  # every preset in a headless browser: chromium (default), firefox or webkit
                          # (after build; needs `npx playwright install chromium firefox webkit`)
npm run record            # regenerate the recorded runs in public/recordings
node scripts/perf.ts      # frame times while training, racing and stepping (after build)
```

On Windows with MSYS2, put `C:\msys64\usr\bin` first on `PATH` for `make`, and
use `make wasm EMCXX="python <emsdk>/upstream/emscripten/em++.py"`. Run the native
test binaries with `C:\msys64\ucrt64\bin` before Git's `mingw64\bin` on `PATH`:
otherwise they load Git's older `libstdc++-6.dll` and crash.

## Deploy

Vercel: import the repository with Root Directory `web`; `web/vercel.json`
has the rest. Vercel cannot
compile the engine, so the built engine (`web/public/engine/ag.{mjs,wasm}`) is
committed. After changing anything in `engine/`, run `make -C engine wasm`
and commit the two files with the change.

GitHub Pages: `.github/workflows/pages.yml` builds everything from source and
publishes `web/dist`. Enable Pages with source "GitHub Actions" in the
repository settings, then run the workflow from the Actions tab.

## How it is built

[product-dev/DESIGN_PHASE1.md](product-dev/DESIGN_PHASE1.md) has the design;
[product-dev/progress.txt](product-dev/progress.txt) records each decision as
it was made. In short: a static graph is compiled once into a flat schedule of
micro-steps, which is both what the executor runs and what the stepper shows.
All memory is allocated at compile time. Nothing in the engine throws; every
error names the node and input it is about.

## Credits

Ideas from micrograd, TensorFlow Playground, CNN Explainer, Transformer
Explainer and Netron. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
