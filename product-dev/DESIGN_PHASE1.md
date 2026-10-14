# Phase 1 Design

High-level design, low-level design, function signatures and data flow for Phase 1 (milestones 1A engine, 1B viewer, 1C live training). [PLAN.md](PLAN.md) says what is built and in what order; this document says how.

Status: Phase 1 implemented, 2026-10-14, in the separate repository https://github.com/ashdev-7/ag. Everything below exists as working code. Where implementation changed the design, the text has been updated and §12 lists each change and why. §11 has the measured results.

## 1. Approach

The brief was to build this the way Andrej Karpathy would. I cannot speak for him; what follows are the habits visible in micrograd, ConvNetJS and his other small projects, applied here.

| Habit | What it means in this design |
|---|---|
| Small and flat | 10 headers, one concept each. Plain structs and free functions. No class hierarchies, no templates in the public surface, no third-party C++ code |
| The simplest thing that is correct | A static graph interpreted in node order. No strides, no views, no dynamic tape, no thread pool |
| One place for each fact | One op table drives shapes, forward, both AD modes, the scalar view and the palette |
| Check against a reference | Every op is tested against finite differences and against PyTorch |
| Overfit one tiny case first | The first thing 1A must do is make a 2-layer MLP fit moons; nothing else is built before that works |
| Readable over fast | Kernels are plain loops. Speed is measured before anything is optimised |

## 2. Decisions that changed from PLAN.md

Working the design through removed several mechanisms the plan had assumed. Each change makes the engine smaller.

| Plan said | Design says | Why |
|---|---|---|
| Engine emits a stream of trace events | The **schedule** is computed once at compile time: a flat list of micro-steps. The UI gets the list up front and asks the engine to run to any index | The graph is static, so the order of events is known before anything runs. No event buffer, no streaming, and scrubbing backwards is exact |
| Tensors have strides and views | Always contiguous; reshape and transpose copy | Tensors are at most 64k elements. Views add aliasing bugs and make every kernel harder to read |
| Dependency-count executor with a ready queue | Reverse node order. `numConsumers` is kept per node so the UI can still show "waiting for 2 gradients" | Node order is already topological. The ready queue matters for multi-threaded dynamic graphs, which this is not |
| Views into WASM memory instead of copies | Statistics are computed in C++ and copied (small). Full tensors are copied only for the selected node | Views are invalidated when memory grows and cannot cross to the page without special headers. A 64k-float copy is cheap |
| Memory allocated and freed per step, shown on a timeline | Every buffer is allocated once at compile time. Liveness (`lastUse`) is computed, so the Phase 2 memory timeline shows logical lifetime | No allocation while running means no leak and no out-of-memory mid-training |
| WASM and native builds give the same trace | Same seed gives the same result within one build; across builds they agree to a tolerance | `exp`, `log` and `tanh` come from different maths libraries and can differ in the last bit |
| Ops throw on divide by zero and log of a non-positive number (current code) | Ops follow floating-point rules and produce inf or nan. The executor detects it and reports the node | Nothing throws in the engine, and "the nan first appeared here" is more useful than an exception |
| Graph IR has a text serialisation in the engine | The saved form is JSON on the web side (`GraphSpec`). The engine is built by `addNode` calls and only has a debug `dump` | One parser, in the language where JSON is native |

## 3. High-level design

```
┌──────────────────────────── page (main thread) ─────────────────────────────┐
│                                                                             │
│  presets ─▶ GraphSpec ─▶ validateSpec ─▶ AppState ◀─ reduce(Action)         │
│                                             │                               │
│        GraphCanvas   Transport   Inspector  │  charts (plain canvas fns)    │
│              ▲           │           ▲      │        ▲                      │
│              └───────────┴───────────┴──────┴────────┘                      │
│                                 │                                           │
│                          EngineClient          one object; never throws     │
│                    request/reply by id, watchdog, restart                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                     postMessage  │  Request / Reply / Frame
┌─────────────────────────────────┴───────────── worker ──────────────────────┐
│  dispatch ─▶ bridge (GraphSpec ─▶ addNode calls, struct packing)            │
│                 │  C API: integer handles, status codes                     │
│  ┌──────────────┴──────────────── WASM ──────────────────────────────────┐  │
│  │ capi ─▶ Session ─┬─ Program (train batch) ─┐                          │  │
│  │                  ├─ Program (eval batch)  ─┼─ Exec ─▶ op kernels      │  │
│  │                  ├─ ParamStore (shared) ───┘                          │  │
│  │                  ├─ Optimizer                                         │  │
│  │                  └─ Dataset, Batcher, Rng                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Repository layout:

```
engine/                     C++20 core, no dependencies
  include/ag/               core rng graph ops program exec optim data scalarize session  (+ capi.h)
  src/                      one .cpp per header, plus bcast.hpp and scalar_rules.hpp (internal)
  tests/                    test_skeleton, test_engine, torch_check.py, wasm_moons.mjs
  bench/                    bench.cpp, built natively and to WASM
  Makefile                  test, torch, bench, wasm, wasm-test, wasm-bench
web/
  src/graph/spec.ts         GraphSpec, validation, build plan, link encoding
  src/graph/presets.ts      the 13 Phase 1 presets
  src/engine/protocol.ts    worker messages
  src/engine/bridge.ts      the only code that touches WASM memory (worker and node)
  src/engine/worker.ts      request dispatch and the training loop
  src/engine/client.ts      EngineClient, Session: timeouts, restarts
  src/views/graph.ts        layout, highlighting, local derivatives, URL state
  src/views/draw.ts         every chart, as plain canvas functions
  src/components/           StepView, TrainView, RaceView, GraphCanvas, Landing, Recorded, ErrorBanner
  scripts/record.ts         recorded runs for the fallback; also checks every preset
  scripts/smoke.ts          every preset in headless Chromium
```

The original scalar engine stays in its own repository, untouched.

## 4. Data flow

### 4.1 Build and compile (once per graph)

```
GraphSpec (JSON)
  │ validateSpec            ids unique, inputs exist, no cycle, fields well-formed
  │ toBuildPlan             topological sort, spec id ⇄ engine node number
  ▼
addNode × N                 arity checked; an input must already exist
  ▼
compile(graph, batch)       validate ▸ infer shapes ▸ requiresGrad ▸ numConsumers
  │                         ▸ buffer offsets ▸ limits ▸ four schedules ▸ lastUse
  ▼
Program                     immutable. nothing after this point can fail on shape or size
  ▼
Session                     train Program + eval Program + ParamStore + Optimizer + Dataset
  ▼
SessionInfo → page          node shapes and the schedule for every pass
```

### 4.2 The schedule

For `d = a*b + a` with a loss on `d`, the Reverse schedule is:

```
0 Forward a      3 Forward d         6 Vjp d slot 1   (d → a, direct)
1 Forward b      4 SeedGrad d        7 Vjp c slot 0   (c → a)   a.grad accumulates here
2 Forward c      5 Vjp d slot 0      8 Vjp c slot 1   (c → b)
```

Each backward micro-step moves gradient along exactly one edge, so the stepper can show two contributions arriving at `a` one after the other. A Train pass appends one `Update` per parameter. A ForwardMode pass is the forward steps, one `SeedTangent`, then one `Jvp` per op node.

### 4.3 Stepping

```
begin(pass)      draw batch, snapshot, zero grads, cursor = 0
advance(n)       run n micro-steps, return the cursor
seek(i)          restore snapshot, replay steps 0..i-1        (exact, because deterministic)
read / stats     copy the selected node's value or grad to the page
```

### 4.4 Training

```
page: run(watch, stepsPerFrame)
worker loop, one tick:
    sessionTrain(stepsPerFrame)              each step is a transaction
    build Frame: loss, stats for watched nodes, tensors for the selected node, grid
    post Frame, then wait for ackFrame       at most one frame in flight
    yield to the message queue               so pause and reset are handled between ticks
page: onFrame → reduce → draw → ackFrame
```

## 5. Low-level design: engine

Full signatures are in the headers; this section gives the shape of each module and the rules it enforces.

### 5.1 `core.hpp`

`Limits`, `Code`, `Status`, `Shape`, `TensorView`, `ConstTensorView`, `Stats`.

```cpp
Status  makeError(Code code, std::string message, NodeId node = kNoNode, int32_t slot = -1);
Status  broadcastShapes(const Shape& a, const Shape& b, Shape* out);
Status  validateShape(const Shape& s, const Limits& limits);
void    computeStats(ConstTensorView t, Stats* out);
int64_t firstNonFinite(ConstTensorView t);
```

Rule: no function in the engine throws. Fallible functions return `Status`, which carries the node and input slot the error is about.

### 5.2 `graph.hpp`

`Op` (23 values, append-only), `Attrs` (one fixed record), `Node`, `Graph`.

```cpp
Status addNode(Graph* g, Op op, std::span<const NodeId> inputs, const Attrs& attrs,
               std::string_view name, std::string_view module, NodeId* outId);
Status setLoss(Graph* g, NodeId node);
Status addOutput(Graph* g, NodeId node);
Status validate(const Graph& g, const Limits& limits);
```

Rule: a node's inputs must have smaller ids. The node list is therefore always in topological order and a cycle cannot be represented.

### 5.3 `ops.hpp`

```cpp
using InferFn     = Status (*)(std::span<const Shape> in, const Attrs& attrs, Shape* out);
using ForwardFn   = void (*)(const ForwardArgs& a);
using VjpFn       = void (*)(const VjpArgs& a);     // += into gin[i] for each bit i of a.need
using JvpFn       = void (*)(const JvpArgs& a);
using ScalarizeFn = Status (*)(const ScalarizeArgs& a);

const OpInfo&           opInfo(Op op);
std::span<const OpInfo> allOps();
const OpInfo*           findOp(std::string_view name);
```

Rule: kernels return `void`. Shapes were proven at compile time, so a kernel has no way to fail except by producing a non-finite number, which the executor checks for after it returns.

| Op | Inputs | Attributes read | Output shape | Local derivative |
|---|---|---|---|---|
| `input`, `param`, `const` | 0 | `shape`, `role`, `init`, `initScale`, `scalar`, `requiresGrad` | `shape` | — |
| `add`, `sub` | 2 | — | broadcast | 1, ±1; summed back over broadcast axes |
| `mul` | 2 | — | broadcast | the other operand |
| `div` | 2 | — | broadcast | 1/b, −a/b² |
| `neg` | 1 | — | same | −1 |
| `pow` | 1 | `scalar` (exponent) | same | n·aⁿ⁻¹ |
| `exp`, `log` | 1 | — | same | out, 1/a |
| `matmul` | 2 | — | [m, k] @ [k, n] → [m, n] | g @ bᵀ, aᵀ @ g |
| `sum`, `mean` | 1 | `axis`, `keepDim` | reduced | g broadcast back (÷ count for mean) |
| `reshape` | 1 | `shape` | `shape` | g reshaped |
| `transpose` | 1 | — | rank 2, swapped | gᵀ |
| `relu`, `leaky_relu` | 1 | `scalar` (slope) | same | 1 if out > 0, else 0 or slope |
| `tanh`, `sigmoid` | 1 | — | same | 1 − out², out(1 − out) |
| `softmax` | 1 | `axis` | same | out ⊙ (g − Σ g ⊙ out) |
| `mse_loss` | 2 | — | scalar | 2(pred − target)/N |
| `softmax_cross_entropy` | 2 | — | scalar | (softmax(logits) − target)/B, computed with log-sum-exp |

### 5.4 `program.hpp`

```cpp
Status                compile(const Graph& g, const CompileOptions& opts, Program* out);
std::span<const Step> schedule(const Program& p, PassKind pass);
bool                  sameParamLayout(const Program& a, const Program& b);
```

`Program` holds a copy of the graph, a `NodePlan` per node (resolved shape, buffer offsets, `requiresGrad`, `numConsumers`, `lastUse`), a `ParamPlan` per parameter and the four schedules. Rule: `compile` builds into a local and assigns to `*out` only on success.

### 5.5 `exec.hpp`

```cpp
Status execInit(const Program* prog, ParamStore* params, Exec* out);
Status execBindInput(Exec* e, NodeId input, std::span<const float> data);
Status execBegin(Exec* e, PassKind pass, TangentSeed seed = {});
Status execAdvance(Exec* e, int32_t n, int32_t* executed);
Status execSeek(Exec* e, int32_t index);
Status execRead(const Exec& e, NodeId node, Which which, std::span<float> out);
Status execStats(const Exec& e, NodeId node, Which which, Stats* out);
```

One arena, sized by `compile`, allocated in `execInit`. The loop is a single `switch` on `StepKind`. After every `Forward` and `Vjp` step the written buffer is scanned; on the first nan or inf the cursor stays put and a `NonFinite` status naming the node is returned and kept in `fault`.

### 5.6 `optim.hpp`

```cpp
Status optimValidate(const OptimConfig& cfg);
Status optimInit(const OptimConfig& cfg, const ParamStore& ps, Optimizer* out);
float  lrAt(const LrSchedule& s, int64_t step);
float  clipScale(float gradNorm, float clipNorm);
void   optimUpdate(Optimizer* o, const ParamPlan& plan, std::span<float> value,
                   std::span<const float> grad, float lr, float scale);
void   optimEndStep(Optimizer* o);
```

State buffers `m` and `v` have the same layout as the parameters, so the UI can show them beside the parameter they belong to. One `optimUpdate` call per `Update` step.

### 5.7 `data.hpp`, `rng.hpp`

```cpp
Status makeDataset(const DatasetConfig& cfg, Dataset* out);
Status nextBatch(Batcher* b, const Dataset& d, int32_t batchSize,
                 std::span<float> xOut, std::span<float> yOut);
```

PCG32 with our own uniform and normal samplers; the standard library's distributions are not used because they differ between toolchains.

### 5.8 `scalarize.hpp`

```cpp
Status scalarize(const Program& p, const Limits& limits, Graph* out, ScalarMap* map);
```

Rewrites the tensor graph into a graph of rank-0 nodes. The same engine then runs it, so the scalar view cannot disagree with the tensor view. Every op has an expansion rule. `ScalarMap::elems[t][e]` is the scalar node holding element `e` of tensor node `t`; reshape and transpose only permute these ids and add no nodes. Matmul, sums and losses expand into chains of scalar `mul` and `add`. Inputs that need no gradient become constants. A graph that would need more than `maxScalarNodes` (500) nodes is refused with `LimitExceeded`; the two-layer MLP preset at batch 4 needs 243.

### 5.9 `session.hpp`

```cpp
Status sessionCreate(const Graph& g, const SessionConfig& cfg, std::unique_ptr<Session>* out);
Status sessionReset(Session* s, uint64_t seed);
Status sessionSetOptimizer(Session* s, const OptimConfig& cfg);
Status sessionBegin(Session* s, PassKind pass, TangentSeed seed = {});
Status sessionAdvance(Session* s, int32_t n, int32_t* executed);
Status sessionSeek(Session* s, int32_t index);
Status sessionTrain(Session* s, int32_t nSteps, TrainStats* out);
Status sessionEvalGrid(Session* s, const GridSpec& grid, std::span<float> out);
Status sessionLossSurface(Session* s, NodeId paramX, int32_t elemX, NodeId paramY, int32_t elemY,
                          const GridSpec& grid, std::span<float> out);
Status sessionReadOptim(const Session& s, NodeId param, int32_t which, std::span<float> out);
Status sessionScalarize(const Session& s, std::unique_ptr<Session>* out, ScalarMap* map);
```

Only a Train pass draws a new batch. Forward, Reverse and ForwardMode passes reuse the current one, so the gradients of the two modes are comparable. Inputs with no data role get their values from their init rule, which is how `a = 2, b = 3` in the first preset are set.

Rule: a training step is a transaction. Parameters, optimizer state and batch position are snapshotted when the step begins and restored if it faults.

How the product features map onto sessions:

| Feature | Implementation |
|---|---|
| Step through forward and backward | `sessionBegin(Reverse)` + `sessionAdvance` / `sessionSeek` |
| Reverse vs forward mode | Same session, `Reverse` once vs `ForwardMode` once per input element; the UI counts passes |
| Decision boundary | `sessionEvalGrid` on the eval program, which shares parameters with the train program |
| Side-by-side runs | Two sessions, same graph and seed, parameters copied with export/import |
| Optimizer race | A preset whose only parameter has shape [2] and whose loss is a test function; one session per optimizer; `sessionLossSurface` draws the background |
| Scalar zoom | `sessionScalarize` returns a second session |

### 5.10 `capi.h`

41 C functions, the only exports of the WASM build (the Makefile exports exactly what `capi.h` declares). Fixed-width structs whose sizes are asserted in the tests, because `web/src/engine/bridge.ts` writes them at fixed offsets. Handles are `generation << 16 | slot + 1`, so a freed or stale handle returns `NotFound`.

## 6. Low-level design: web

### 6.1 `graph/spec.ts`

```ts
validateSpec(spec: unknown): SpecIssue[]
toBuildPlan(spec: GraphSpec, opNumbers: Map<OpName, number>): BuildPlan
encodeSpec(spec: GraphSpec): string
decodeSpec(encoded: string): GraphSpec | SpecIssue[]
```

Nodes are named by string ids that survive edits. Engine node numbers exist only inside the worker; every error and every step comes back to the page with the string id.

### 6.2 `engine/protocol.ts`

Eighteen request types, one reply each, matched by id. Two unprompted messages: `Frame` and `Stopped`. Typed arrays are transferred, not copied.

### 6.3 `engine/client.ts`

```ts
createEngineClient(options: EngineClientOptions): EngineClient

interface EngineClient {
  status: EngineStatus                      // loading | ready | restarting | unavailable
  ready(): Promise<Result<void>>
  createSession(spec: GraphSpec, config: SessionConfig): Promise<Result<Session>>
  dispose(): void
}

interface Session {
  begin(pass, seed?) / advance(n) / seek(index): Promise<Result<Cursor>>
  run(options: RunOptions) / pause() / reset(seed) / setOptimizer(optim)
  read(nodeId, which) / stats(nodeIds, which) / evalGrid(grid) / lossSurface(...)
  exportParams() / importParams(params) / scalarize() / destroy()
}
```

Rule: every method resolves to `Result<T>`. Nothing in the client throws or rejects, so no view needs a try/catch.

### 6.4 Views

`App` holds what every view shares: the engine client and its status, the preset and seed from the URL, and the one error banner. Each mode is one component that owns its own state (`StepView`, `TrainView`, `RaceView`); when the engine restarts, the view is remounted and rebuilds its sessions from spec and seed. Charts are plain functions that draw onto a canvas context (`views/draw.ts`: `drawTensor`, `drawField`, `drawPoints`, `drawHistogram`, `drawLines`, `drawPaths`). `toUrlHash` / `fromUrlHash` carry the shareable part of the state: preset, seed, pass, step, zoom, optimizers.

Frontend stack: Vite, TypeScript, React, React Flow for the graph, dagre for layout, Canvas 2D for every chart. No state library and no chart library.

## 7. Fault model

| Fault | Detected by | Response | What the user sees |
|---|---|---|---|
| Malformed spec or shared link | `validateSpec`, `decodeSpec` | Nothing is sent to the engine | Message naming the node; the previous graph stays |
| Cycle in the graph | `toBuildPlan` | As above; the engine cannot represent one | "These nodes form a loop" |
| Wrong arity or shape | `addNode`, `compile` | `Status` with node and slot; `out` untouched | The offending edge is marked with both shapes |
| Model too large | `compile` against `Limits` | `LimitExceeded` naming the limit | Which limit, its value, and why it exists |
| nan or inf in a forward value or gradient | Scan after each step | Step stops; training step rolled back to its snapshot | Training pauses, the node where it first appeared is marked, parameters are the last good ones |
| Bad optimizer settings | `optimValidate` | Rejected; old optimizer kept | Field-level message |
| Call in the wrong order | `InvalidState` | No state change | Should not be reachable from the UI; logged |
| Stale or freed handle | Handle generation check | `NotFound` | Session is recreated |
| Engine invariant broken | `Internal` status | Session marked dead and recreated from the spec | "Engine error, run was reset", with a report link |
| WASM trap (out-of-bounds, stack overflow, abort) | Worker `error` event | Client rejects pending calls with `WorkerCrashed`, restarts the worker, recreates sessions from spec and seed | Banner; training restarts from step 0 |
| Worker hung | Client watchdog (5 s without a reply) | Terminate and restart as above | Same |
| Repeated crashes | Restart counter (2 in a row) | Status `unavailable` | Recorded preset is shown; controls that need the engine are disabled |
| WebAssembly unavailable or fails to load | `ready()` | Status `unavailable` | Same as above |
| Page slower than training | Frame acknowledgement | At most one frame in flight; the frame rate falls, training does not | Smooth but less frequent updates |
| Tab hidden | `visibilitychange` | Training pauses | Resumes where it stopped |
| Unbounded history | Loss history thinned beyond a fixed length | Memory stays flat on long runs | — |

Three properties make most of this table short:

1. **Everything is checked at compile time.** After `compile` succeeds, the only runtime failure is a non-finite number.
2. **Nothing is allocated while running.** One arena per executor, sized up front.
3. **State can always be rebuilt.** Spec, seed and step count reproduce any run, so the recovery for every serious fault is the same: recreate the session.

The WASM build uses a fixed-size heap with growth disabled and exceptions disabled.

## 8. Cross-check against existing tools

Read on 2026-10-14 from each project's source: file lists, and the specific functions named below. This is a reading of structure, not a full audit.

| Tool | What its code does | What this design takes or does differently |
|---|---|---|
| TensorFlow Playground | Five source files. The network is arrays of `Node` and `Link` with free functions `forwardProp`, `backProp`, `updateWeights` (`nn.ts`). The decision boundary runs `forwardProp` over a 100 × 100 grid (`DENSITY = 100`) and draws it one pixel per sample on a canvas that CSS scales up. All settings, including the seed, are serialised into the URL hash (`state.ts`). Training runs on the main thread in a timer | **Taken:** the scale of the codebase; grid of 100; one-pixel-per-sample heatmap (`drawField`); URL-hash state with seed. **Different:** compute is in a worker, so the page cannot freeze; a diverged run is rolled back and reported instead of filling the display with nan |
| CNN Explainer | Loads a pretrained model with TensorFlow.js, runs it once, then builds its own `Node`/`Link` graph from all layer outputs (`constructCNNFromOutputs`). Views are an overview plus detail views with step animators | **Taken:** draw from a complete snapshot, never from a half-updated engine (`Frame`); overview plus inspector. **Different:** our snapshot comes every few steps of live training |
| Transformer Explainer | Runs GPT-2 with ONNX Runtime Web (WASM), fetched as 63 chunks with caching. Ships five precomputed examples, each about 1.6 MB, that the page can show without the model | **Taken:** recorded results per preset, so the page shows something before the engine loads and when it cannot run (`unavailable` status). **Different:** our recordings are small because the networks are |
| Netron | One model abstraction (`view.Graph`, `view.Node`, `view.Value`) independent of file format. Graph layout runs in a worker that can be cancelled and terminated. UI actions are wrapped so that every failure goes to one `error()` screen | **Taken:** one error surface (`banner`); a client that owns a worker and can terminate it; string-id graph model separate from the engine's numbering. **Different:** layout stays on the main thread at our 200-node cap; to be revisited if it is slow |

The comparison changed four things in the design: the grid default of 100, the recorded-preset fallback, the single error banner, and snapshot-per-frame drawing.

## 9. Testing

| Level | What | Reference |
|---|---|---|
| Op | Forward value, VJP and JVP for every op at random points | Central finite differences in double precision |
| Op | VJP and JVP agree: ⟨v, J·u⟩ = ⟨Jᵀ·v, u⟩ for random u, v | Each other |
| Op | Same values and gradients | PyTorch, via golden files generated by a Python script |
| Optimizer | Ten steps on a fixed problem | `torch.optim` |
| Compile | Every row of the fault table that compile owns | Expected `Code`, node and slot |
| Session | Seek to i equals advancing i times; a faulted step leaves parameters unchanged; reset reproduces a run | Itself |
| Session | No arena growth over 10,000 steps | Arena size |
| Scalar view | Scalar session gradients equal tensor session gradients | Each other |
| End to end | 2-layer MLP reaches a target loss on moons from a fixed seed | Fixed threshold |
| Web | `validateSpec` and `fromUrlHash` on malformed input never throw | — |
| Web | Client recovers when the worker is killed mid-run | — |

Every row of this table is implemented. The PyTorch comparison reads cases the C++ test writes (`--dump`) instead of golden files, so the two never drift apart. The browser smoke test (`web/scripts/smoke.ts`) was added on top: it drives every preset in headless Chromium and fails on any console error.

## 10. Build order, as it went

1. `d = a*b + a`: d = 8, ∂d/∂a = 4, ∂d/∂b = 2, through the tensor engine, the stepper and the C API. Stopping after step 7 shows a.grad = 1, the direct contribution only.
2. Finite differences and forward-vs-reverse agreement for all 23 ops and a two-layer MLP: worst relative error 5e-5.
3. Fit moons with SGD: val loss 0.048, val accuracy 80 of 80.
4. PyTorch: all 26 op cases agree to 3e-7, all six optimizers over 10 steps to 2e-5.
5. Sessions: seek equals advance (this test found that `execBegin` must clear values, see §12), rollback, reset reproducibility, no arena growth over 10,000 steps.
6. Emscripten: the WASM build matches the native batch losses over 1,500 steps to 1.4e-7.
7. Web: bridge, worker, client, views. Every preset was run through the engine by `scripts/record.ts` before any view was built; that found that the spirals preset did not learn with tanh (§12).
8. Browser smoke test, screenshots inspected; found and fixed a graph-fit bug and two near-identical chart colours.

## 11. Results

| Check | Result |
|---|---|
| `make test` (native) | 171 checks pass (154 at the end of the audit, 17 for detach and no_grad), plus 33 in the skeleton test; g++ and clang (Emscripten) both warning-free; the same suite runs clean with undefined-behaviour traps on (`-fsanitize=undefined,float-cast-overflow -fsanitize-trap=all`) |
| `make torch` | 26 op cases and 6 optimizers match PyTorch 2.9 |
| `make wasm-test` | WASM agrees with native to 1.4e-7 after 1,500 steps; errors carry spec ids |
| CMake build + ctest | both test binaries pass |
| `npm test` | 42 tests: spec validation, URL parsing, client recovery (hung worker, trapped worker, repeated crashes, failed load) and frame pacing; every local derivative the inspector prints against finite differences; colour separation under simulated colour blindness; backward order; colour scales |
| `npm run build` | 135 KB of JS gzipped, plus a 170 KB engine |
| `scripts/smoke.ts` | 61 checks in each of Chromium 131, Firefox 132 and WebKit 18.2: all 14 presets, every edge drawn, the comparison link, a link to a step, keyboard control, phone layout, no console errors |
| `scripts/perf.ts` | see §14.3 |

Step time (µs per training step, Adam, batch 32 unless noted):

| Model | Params | Native (g++ -O2) | WASM (node) |
|---|---|---|---|
| 2 × 8 tanh | 112 | 49 | 66 |
| 2 × 16 | 352 | 90 | 166 |
| 2 × 32 | 1,216 | 207 | 407 |
| 2 × 64 | 4,480 | 610 | 1,357 |
| 2 × 128 | 17,152 | 2,400 | 4,329 |
| 4 × 64 | 12,800 | 1,779 | 4,410 |
| 2 × 64, batch 128 | 4,480 | 2,388 | 5,633 |
| 4 × 128, batch 128 | 50,176 | 24,630 | 42,834 |

In the browser the presets train at 300 to 650 steps per second at the default 10 steps per frame; the limit there is the frame round trip, not the engine.

## 12. Changes made while implementing

| Design said | Built | Why |
|---|---|---|
| `execBegin` zeroes grads and tangents | It also zeroes every computed value | Found by the seek test: values left from an earlier pass made "seek to i" differ from "advance i" in nodes not yet reached. Now the state after k steps depends on nothing else, and the stepper shows unreached nodes as empty |
| `ScalarMap::first`: a tensor's scalars are contiguous | `ScalarMap::elems`: one id per element | Matmul and sum chains end in nodes that are not contiguous; reshape and transpose become free |
| `sessionLossSurface(param, elemX, elemY)` | `(paramX, elemX, paramY, elemY)` | The race presets use two scalar parameters x and y, and the engine has no slice op |
| 37 C functions | 41: added `ag_session_node_op`, `ag_session_node_name` (to draw the scalar graph), `ag_session_dataset` (to draw the data points) and `ag_session_read_optim` (the optimizer-state view) | Each was needed by a Phase 1 view |
| `fillInit` was internal | Declared in `exec.hpp` | Inputs with no data role are filled by the same rule as parameters |
| One `AppState` with `reduce` | Shared state in `App`, the rest in each view | The three modes share almost nothing; one reducer would have been a switch over three unrelated states |
| CMake as the build | A Makefile as the primary build; CMake kept | One file covers native, PyTorch, WASM and benchmarks with no generator; CMake still builds and runs the tests |
| Spirals: 2 × 16 tanh | 2 × 32 ReLU, batch 32 | Tanh networks sat on a plateau (val loss 0.65 after 4,000 steps); ReLU 32-32 reaches 0.03 |
| Random draws for the datasets | Each draw in its own statement | Two draws in one argument list run in an order that differs between g++ and clang, which would have made the WASM dataset differ from the native one |
| `requires_grad`, `no_grad`, `detach` (PLAN 1A) | `requiresGrad` on inputs and parameters only | Nothing in Phase 1 needed the other two. Built afterwards, see §14.1 |
| Module zoom level | Not built | PLAN lists scalar and tensor zoom for Phase 1; `module` is set on preset nodes for Phase 2 |

## 13. Audit fixes

An audit after Phase 1 looked for overflow, memory exhaustion, graph building errors and state bugs. Each bug below was reproduced first (`test_engine.cpp` `[robustness]`, `wasm_moons.mjs`, `smoke.ts`) and now has a test.

| Bug | Effect | Fix |
|---|---|---|
| `numel` overflowed for shapes like `[65536]×4` (2⁶⁴ wrapped to 0) | The shape passed the size limit and compiled with a zero-size buffer; kernels on it would read and write past the arena | `validateShape` checks the running product against the limit as it multiplies; `numel` saturates instead of wrapping |
| Histogram bin computed as `int((x − lo) / width · 32)` in float | For values near ±3e38 the width overflowed to inf, the position became nan, and `int(nan)` is undefined (it can trap in WebAssembly) | Computed in double, clamped before the conversion |
| Doubles converted to float without a range check (sums, means, losses, norms) | Out-of-range conversion is undefined behaviour; on a diverging run the executor might not see inf | `toFloat` in `core.hpp` saturates to ±inf; used at every such conversion |
| Non-finite attributes accepted | A nan exponent or init scale surfaced later as a nan somewhere else | `validate` refuses them, naming the node |
| Uniform init chose its branch by the sign of the bound | A negative or zero uniform scale silently initialised to all zeros | The init kind decides uniform or normal; the width is taken as an absolute value |
| Out of memory in the WASM build | The fixed 128 MB heap ran out inside `std::vector`, which with exceptions off aborts the module. A malloc-then-free probe did not help: the compiler may delete an unused malloc/free pair | The arena is a `FloatBuffer` allocated with `calloc` and kept, and the build uses `ABORTING_MALLOC=0`; a full heap is now an `OutOfMemory` error and the engine keeps working |
| `sessionAdvance` rolled back on any error in a training pass | An error such as "pass not begun" restored parameters from a stale snapshot, undoing real training | Rolls back only on `NonFinite` |
| Changing the optimizer or importing parameters mid-way through a stepped training step | The rollback point was gone; seeking or a later fault could leave the step's updates applied | The step is cancelled; continuing or seeking it is refused until a new begin |
| No node limit in `ag_graph_add_node`; handle slots unbounded | A runaway caller could fill the heap; a 65,536th slot would alias handle slot numbers | Graphs stop at the node limit; the handle table stops at 65,535 slots |
| Bridge: a session created in the engine leaked if reading its plan failed; same for the scalar view | Engine memory leak per failure | `adopt()` frees the handle on failure |
| Bridge: training step's low 32 bits read as signed | Negative step counts past 2³¹ steps | Read unsigned |
| Bridge: decision grid for a model with one input left part of the buffer unwritten | Uninitialised memory drawn | The buffer is filled with nan first |
| Worker: `maxSteps` counted the run, not the session | A resumed race overshot its step limit | The run starts from the session's real step count |
| Stepper: switching zoom in forward mode started a pass seeded at a node of the other graph | A false "no node" error banner | The pass waits for a seed that belongs to the current graph |
| Race view: button stayed on "Pause" after every racer finished; raced on in a hidden tab | Wrong state, wasted work | Counts stopped racers; pauses when the tab is hidden |

The audit also checked that the single-number graph builds and matches the tensor graph for every op case, including edge cases (an input used twice, `x @ xᵀ`, rank-4 broadcasting, keepdim, reshape to a scalar); all passed, so no graph-building bug was found there.

## 14. Solidifying Phase 1

Work after the audit, on the gaps between Phase 1 as built and PLAN/PRD: E10, V8, the accessibility and browser requirements of PRD §9, and the frame budget of PLAN §5.

### 14.1 `detach` and `no_grad` (PRD E10)

| Piece | Built |
|---|---|
| Graph | `Op::Detach` appended to the op table, so existing op numbers hold; `Attrs::noGrad` on any computed node (refused on sources) |
| Compile | `requiresGrad` stops at both. The loss's gradient travels only through nodes that need one, so a parameter seen only through a detach gets no buffer, no contributions and no update step |
| Both AD modes and the scalar view | Stop at the same place; the scalar view emits one detach per element and marks every computed piece of a no_grad node |
| C API | `ag_attrs.no_grad`; the struct grows from 48 to 52 bytes and `AG_ABI_VERSION` becomes 2, so a cached old engine is refused by a new page |
| Web | `detach` and `noGrad` in GraphSpec; a `detach` stepping preset (`d = detach(a)·b + a`, so a's gradient is 1, not 4); edges into nodes where the gradient stops are dashed and the node says "no ∂" |
| Tests | Reverse, forward and scalar views give the stopped gradient; an untrained detached parameter under Adam; a no_grad loss trains as a no-op; no_grad on a parameter is refused at that node; both cases compared with PyTorch (`.detach()` on the node's inputs) |

### 14.2 Backward order and the ready queue (PRD V8)

During a Backward or Train pass the inspector lists the nodes in the order they pass gradient back, then the leaves. Each is waiting (with k of n contributions when it has several uses), ready (complete, not yet passed on: PyTorch's ready queue), passing back, or done. Over 40 nodes it shows counts instead of the list. `backwardQueue()` in `views/graph.ts` is a pure function of the schedule and the cursor.

### 14.3 Frame budget

`scripts/perf.ts` measures, in a real browser, the interval between animation frames while each training preset runs at 10 and 100 steps per frame and while each race runs, and the time from a key press to the next step on screen.

The first measurement failed the budget: at the default speed 17–40% of frames took 33 ms. The client acknowledged each frame on receipt, so a fast worker kept the page redrawing back to back, and each redraw re-rendered the React Flow diagram (hiding it halved the late frames). Now frames are acknowledged on the next display frame, the diagram updates at most ten times a second while training, and `GraphCanvas` is memoised.

Chromium 131, headless, on a machine running about 3× slower than when §11's benchmark was taken (the WASM benchmark under node moved from 407 to 1,268 µs per step for 2 × 32 in this session, so compare rows here with each other, not with §11):

| Run | p95 frame | Late frames | Steps/s |
|---|---|---|---|
| Training presets, 10 steps per frame | 16.7–16.8 ms | 0–3% (was 17–40%) | 337–564 |
| Training presets, 100 steps per frame | 16.7–16.8 ms | 0–2% | 1,391–5,132 |
| Races | 16.7–16.8 ms | 0–1% | |
| One step, MLP as tensors (13 nodes) | 23 ms | | p50 10 ms |
| One step, MLP as single numbers (243 nodes) | 68 ms | | p50 48 ms |

The cost: at 100 steps per frame throughput is lower than before (moons 3.9k against 5.8k steps/s), since the worker no longer runs ahead of the screen. The earlier figure came with dropped frames.

Other browsers, same machine: Firefox 132 holds its frames (headless Firefox runs at 144 Hz, so its late-frame share is not comparable) but trains 2–4× slower, because the engine itself is 2–4× slower there (the WASM benchmark in the page: 5,550 against 1,306 µs per step for 2 × 32). That is Playwright's patched Firefox; no stock Firefox was available to confirm it. WebKit 18.2 on Windows draws an idle page at 32 fps, so its frame numbers say nothing about Safari; its engine speed is within 15% of Chromium.

### 14.4 Accessibility (PRD §9)

| Requirement | Done |
|---|---|
| Colours that do not rely on red against green | Checked under simulated protanopia, deuteranopia and tritanopia (Machado 2009). Three pairs failed: two racer colours (ΔE 6.8), two layer colours (5.8), tangent against forward (9.7). Now four series colours with every pair over ΔE 25; layers on an indigo-to-orange ramp whose lightness rises with depth for every kind of colour vision; tangent #882255; paths on the loss surface over a light halo; the mode comparison says "They agree"/"They differ" instead of green against red. `colors.test.ts` reads the real values and keeps this checked |
| Keyboard control of the stepper | ← → step, space plays, Home and End jump; ignored while typing in a field or on a focused button. Transport buttons have labels, toggles `aria-pressed`, the step counter is announced, focus is visible |

### 14.5 Bugs found on the way

| Bug | Effect | Fix |
|---|---|---|
| Graph nodes had no `measured` size | React Flow drops handle positions for such nodes on every new node array, and edges are drawn only between measured handles, so about one load in three showed a graph with no edges, for good | Pass the layout's size as `measured`; the smoke test counts drawn edges for every stepping preset |
| A link with `i=N` opened at step 0 | Setting the default forward-mode direction restarted the pass, and the superseded run had already used up the linked step | Only the run that is still current uses it |
| Frames acknowledged on receipt | See §14.3 | Acknowledged on the next display frame |

Not a code bug, but it cost time: on this machine Git for Windows' `/mingw64/bin` is on `PATH` before MSYS2's `ucrt64/bin`, so the native test binary loaded an older `libstdc++-6.dll` and crashed while writing its PyTorch dump. With `C:\msys64\ucrt64\bin` first it passes.

