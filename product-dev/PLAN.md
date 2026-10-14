# Development Plan

A two-phase plan to turn this autograd engine into a browser product where people build neural networks by dragging and connecting nodes, then watch every number move during training. The companion document is [PRD.md](PRD.md), which says what the product is and for whom; this document says how it gets built.

Status: revised 2026-10-14. Phase 1 is implemented in the repository https://github.com/ashdev-7/ag (milestones 1A, 1B and 1C; deployment is a configured workflow waiting for GitHub Pages to be switched on). The design, its results and the changes made while building are in [DESIGN_PHASE1.md](DESIGN_PHASE1.md). Phase 2 has not started.

## 1. The central idea

The engine emits a step-by-step **trace** of everything it does: each forward op, each gradient accumulation, each tensor saved or freed, each optimizer update. The website is a viewer and editor for that trace. This makes the product a data-flow diagram of neural network training, and it keeps the engine and the interface independent: any feature is "emit a new event" plus "draw that event".

## 2. Starting point

The repository today is micrograd in C++ with Python bindings.

| File | Contents |
|---|---|
| `include/Tensor.hpp`, `src/Tensor.cpp` | `TensorNode` (data, grad, inputs, parents, `backwardFn`), recursive topological sort, `backward()` |
| `include/Ops.hpp`, `src/Ops.cpp` | 8 ops: add, sub, mul, div, pow, exp, log, relu |
| `src/Bindings.cpp` | pybind11 module `autograd` exposing `Tensor` |
| `tests/test_engine.cpp` | 13 test functions with hand-derived expected gradients |

What blocks the product, all addressed in milestone 1A:

| Problem | Where | Consequence |
|---|---|---|
| Scalar only | Every op reads and writes `data[0]` and `grad[0]` | No shapes, matmul, broadcasting or batches |
| Opaque backward closures | `backwardFn` is a `std::function<void()>` | Cannot be stepped, serialised, listed in a palette or run in forward mode |
| Self-reference leak | Each closure captures `out`, and `out` owns the closure | Op nodes are never freed; a training loop grows without bound |
| Recursive `topoSort` | `src/Tensor.cpp` | Stack overflow on deep graphs |
| Intermediate grads never zeroed | `backward()` | A second backward call compounds gradients on interior nodes |
| Op nodes always `requiresGrad = true` | `makeNode` | Gradient is tracked through frozen subgraphs |
| Tests cannot fail the build | `main()` returns 0; no finite-difference check | Regressions go unnoticed |
| No optimizer, layers or losses | — | Nothing can train |

## 3. What we take from existing work

### 3.1 Engine design: micrograd and PyTorch

| Source | What it does | What we do |
|---|---|---|
| micrograd `engine.py` | One `Value` per scalar; a `_backward` closure per node; DFS topological sort | Keep the scalar view as a *zoom level* for teaching, generated from the tensor graph rather than executed separately |
| micrograd `nn.py` | `Module` → `Neuron` → `Layer` → `MLP`, with `parameters()` | Same module hierarchy, used for grouping nodes on the canvas |
| PyTorch `engine.cpp` | Counts each node's dependencies, runs nodes from a ready queue when the count reaches zero, accumulates incoming gradients in an input buffer | Adopt this executor. The ready queue and the accumulation are themselves visible steps |
| PyTorch saved variables | Each node saves the tensors its backward needs and releases them after it runs unless `keep_graph` is set | Explicit save and release events, which drive the memory timeline |
| PyTorch `checkpoint.py` | Runs a segment without recording, keeps only its inputs, recomputes the segment during backward | Implement the same mechanism for a checkpoint region drawn on the canvas |
| PyTorch forward-mode AD | Pushes tangents forward alongside values | Every op declares a JVP next to its VJP |

### 3.2 Existing tools: what to reuse, port or only learn from

Licence, stars and last push were read from the GitHub API on 2026-10-14. What each tool does is from its repository description and search summaries; the apps themselves were not each opened and tested.

| Project | Licence | Last push | What it is | How we use it |
|---|---|---|---|---|
| [xyflow / React Flow](https://github.com/xyflow/xyflow) | MIT | 2026-10 | Node-based editor library | **Depend on it** for the canvas |
| [dagre](https://github.com/dagrejs/dagre) | MIT | 2026-08 | Directed graph layout | **Depend on it** for automatic layout |
| [TensorFlow Playground](https://github.com/tensorflow/playground) | Apache-2.0 | 2026-06 | Dense networks on 2D toy data, live training | **Port** the decision-boundary heatmap, loss chart and dataset generators; copy its "edge thickness shows weight" idea |
| [ENNUI](https://github.com/martinjm97/ENNUI) | MIT | 2023-01 | Drag-and-drop builder, trains in browser, exports Python | **Study and port** palette design, shape checking and code export. Closest to our builder, but stale and not built on a transparent engine |
| [CNN Explainer](https://github.com/poloclub/cnn-explainer) | MIT | 2023-10 | One CNN, animated forward pass | **Port ideas** for the convolution and pooling views |
| [Transformer Explainer](https://github.com/poloclub/transformer-explainer) | MIT | 2026-06 | GPT-2 in the browser, forward pass | Model for the attention view |
| [llm-viz](https://github.com/bbycroft/llm-viz) | MIT | 2026-08 | 3D walkthrough of a GPT-style model | Model for guided, step-by-step narration |
| [Netron](https://github.com/lutzroeder/netron) | MIT | 2026-10 | Viewer for saved model files | Model for rendering large graphs legibly |
| [ConvNetJS](https://github.com/karpathy/convnetjs) | MIT | 2023-01 | In-browser training demos | Model for small live-training demos |
| [micrograd](https://github.com/karpathy/micrograd) | MIT | 2026-08 | Scalar autograd engine | Model for the scalar zoom level |
| [Distill, "Why Momentum Really Works"](https://github.com/distillpub/post--momentum) | CC-BY-4.0 | 2019-08 | Interactive optimizer article | Model for the optimizer race |
| [loss-landscape-anim](https://github.com/logancyang/loss-landscape-anim) | MIT | 2024-01 | Python: optimizer paths on a 2D slice of a loss landscape | Its method (PCA of the path to choose the slice) for showing a real network's landscape |
| [autograd-ui](https://github.com/hewliyang/autograd-ui) | none | 2024-01 | Scalar autograd graph viewer in the browser | Closest to our stepper. **Ideas only**: no licence means no right to copy |
| [neural-network-from-scratch](https://github.com/Ameobea/neural-network-from-scratch) | none | 2024-10 | Rust engine compiled to WASM with a web visualiser | Closest to our architecture. **Ideas only** |

Decision: **do not fork any of these applications.** None of them exposes autograd at the level of individual ops, and the two builders that exist run on TensorFlow.js, which hides exactly what this product is meant to show. The engine and its trace stay ours. We build on open source in two ways: as libraries (React Flow, dagre) and by porting specific, well-proven views from the MIT and Apache-2.0 projects.

Obligations when porting: MIT code keeps its copyright notice; Apache-2.0 code keeps its notice and a statement of what we changed. A `THIRD_PARTY_NOTICES` file is added the first time anything is ported. Repositories with no licence are all rights reserved, so nothing is copied from them.

## 4. Architecture

```
 Browser main thread                      Web Worker
 ┌──────────────────────────┐            ┌─────────────────────────────┐
 │ Canvas editor (graph IR) │──graph──▶ │ C++ core compiled to WASM   │
 │ Stepper / timeline       │◀──trace───│  op registry                │
 │ Tensor and chart views   │◀──views───│  executor (forward, VJP,JVP)│
 └──────────────────────────┘            │  optimizers, data, RNG      │
                                         └─────────────────────────────┘
```

The detailed Phase 1 design refined items 4 to 6 and the trace events below: the trace became a schedule computed at compile time, the executor walks nodes in reverse order, and tensors are copied to the page on request. [DESIGN_PHASE1.md](DESIGN_PHASE1.md) section 2 lists each change and why; where the two documents differ, that one is current.

Decisions:

1. **C++ core, compiled to WebAssembly with Emscripten.** All compute runs in the user's browser, in a Web Worker so the interface never blocks. No compute server, so the site is static.
2. **Graph IR is separate from execution.** The user edits a static description (nodes, edges, parameters). The engine executes it and reports back. The same IR is what gets saved, shared and exported.
3. **Op registry replaces closures.** Each op is a record: name, arity, `shape_infer`, `forward`, `vjp`, `jvp`, and which tensors it needs saved. The node palette, both AD modes, validation and code export all read from this one table.
4. **Dependency-count executor** in place of recursive DFS.
5. **Trace is binary, not JSON.** Events are fixed-size records in a buffer; tensor contents are read as views into WASM memory. Copying every tensor across the boundary as JSON would dominate step time.
6. **Trace levels.** Full (every event, for stepping), summary (per-layer statistics, for live training), off (for speed). The interface picks the level.
7. **Three zoom levels.** Scalar, tensor and module. Scalar view is offered only for small tensors.
8. **Deterministic.** Seeded RNG in the engine so a shared link reproduces the same run.
9. **Python bindings stay as a test oracle** that compares gradients against PyTorch. They are not part of the product.
10. **Frontend: TypeScript, React, React Flow for the canvas, Canvas/WebGL for heatmaps and surfaces.** Default choice, open to change before Phase 2 starts.

### Trace events (first sketch)

| Event | Carries |
|---|---|
| `forward` | node id, output tensor handle, shape |
| `save` / `release` | node id, tensor handle, bytes |
| `ready` | node id entering the backward ready queue |
| `accumulate` | target node, source node, incoming gradient handle |
| `vjp` / `jvp` | node id, incoming and outgoing gradient or tangent handles |
| `param_update` | parameter id, old value, update, new value handles |
| `opt_state` | parameter id, buffer name (momentum, second moment), handle |
| `step_end` | step number, loss, bytes live |

## 5. Constraints

Revised after measuring in milestone 1A (DESIGN_PHASE1.md §11 has the full benchmark). The engine enforces each limit in `ag::Limits`.

| Limit | Budget | Measured (WASM, under node) | Reason |
|---|---|---|---|
| Parameters per model | 100k (enforced) | 17k params, batch 32: 4.3 ms per step; 50k params, batch 128: 43 ms | Keeps a train step inside one frame up to about 20k parameters at batch 32. Larger models still train, a few steps per frame |
| Elements per tensor | 64k (enforced) | — | Readable as a heatmap; cheap to copy |
| Nodes in a graph | 256 (enforced) | — | Layout and legibility |
| Scalar view | 500 scalar nodes (enforced) | The 2-4-2 MLP at batch 4 needs 243 | Beyond that the graph is unreadable |
| Train step | under 16 ms | Every Phase 1 preset: 0.07 to 0.4 ms | Training animates at 60 fps with many steps per frame |
| Working memory | 8M floats per executor (enforced); 128 MB fixed WASM heap | — | No allocation while running; out of memory is a trap the client recovers from |
| Numerics | float32, CPU, single thread, no SIMD yet | WASM runs 1.4 to 2.3 times slower than native g++ -O2 | Plain loops were fast enough; SIMD and threads only if a Phase 2 preset needs them |
| Datasets | 2D toys (moons, spirals, XOR, circles), 1D regression, a small-image MNIST subset, tiny character sequences | Small enough to ship with the page |
| No GPU | — | WebGPU is a possible later addition, not part of this plan |

What this rules out: ResNet-50, real image datasets, large transformers. "ResNet" in this product means 3 to 6 residual blocks on small data, which is enough to show why skip connections help.

## 6. Syllabus coverage

| Topic | What it does | What the user sees | Phase |
|---|---|---|---|
| Computation graph, chain rule | Records ops and multiplies local derivatives | Values and gradients on every edge, one step at a time | 1 |
| Reverse vs forward mode | One pass per output vs one pass per input | The same graph run both ways, with pass counts | 1 |
| Tensors, shapes, broadcasting | Batched maths; broadcast gradients sum back | Shape labels, heatmaps, the sum-back step | 1 |
| Activations (ReLU, leaky ReLU, tanh, sigmoid, GELU, softmax) | Non-linearity | The curve, its derivative and where each unit sits on it | 1 |
| Losses (MSE, cross-entropy) | Scalar training target | Per-sample loss and the gradient it seeds | 1 |
| Initialisation (Xavier, He) | Keeps signal scale stable through depth | Per-layer activation histograms at step 0 | 1 |
| Optimizers (SGD, momentum, Nesterov, RMSProp, Adam, AdamW) | Turn gradients into parameter updates | Paths on a loss surface; internal state buffers | 1 |
| Learning-rate schedules | Change step size over time | The schedule curve against the loss curve | 1 |
| Pathologies (vanishing, exploding, dead ReLU) and gradient clipping | Why training stalls or diverges | Per-layer gradient norms over time | 1 |
| Layer sizes, depth and width | Capacity | Decision boundary changing as layers are resized | 2 |
| Connections (skip, concat, weight sharing) | Gradient highways and parameter reuse | Gradient flow with and without the skip | 2 |
| Regularisation (dropout, weight decay) | Limits overfitting | Dropout masks; train vs validation gap | 2 |
| Normalisation (batch, layer) | Stabilises activation scale | Distributions before and after the layer | 2 |
| Convolution, pooling | Local, shared weights | Kernel sliding over input; feature maps | 2 |
| Memory (saved activations, `no_grad`, `detach`, checkpointing) | Trades compute for memory | Memory timeline; checkpointed segments recomputing | 2 |
| Attention (single head) | Content-based mixing | Attention matrix and its gradients | 2, stretch |

Out of scope: large transformers, GANs, diffusion, reinforcement learning, mixed precision, distributed training.

## 7. Phases

Development is two phases. Each phase ends in a public release and is split into three milestones, each of which ends in something that runs and is signed off before the next begins.

### Phase 1 — Glass: see a network learn

Goal: a public site where a user picks a preset network, steps through the forward and backward pass number by number, then trains it live and compares optimizers. No editing yet.

**Milestone 1A — Engine**

- Tensor type: float32, contiguous, rank up to 4; reshape and transpose copy.
- Op registry and the first op set: add, sub, mul, div, neg, pow, exp, log, matmul, sum, mean, reshape, transpose, broadcast, relu, leaky relu, tanh, sigmoid, softmax.
- Broadcasting with gradient sum-back.
- Compiler that checks shapes and limits and builds the schedule; executor that walks it; reverse mode (VJP) and forward mode (JVP).
- All buffers allocated once at compile time, with computed liveness; closures removed, which fixes the leak.
- `requires_grad` propagation, `no_grad`, `detach`.
- Graph IR built by calls, saved as JSON on the web side; schedule of micro-steps per pass; per-node statistics.
- Non-finite detection with rollback of the faulting training step.
- Seeded RNG.
- Emscripten build target alongside the native build; a minimal worker harness.
- Tests: a runner with a failing exit code, finite-difference checks for every op, VJP against JVP consistency, comparison against PyTorch through the Python bindings.
- Benchmark: step time against model size, native and WASM.

Done when: a 2-layer MLP trains on moons headlessly with a hand-written SGD update; every op passes finite-difference and PyTorch comparison; no memory growth across 10,000 steps; the WASM and native builds agree within tolerance for a fixed seed; the constraint table in section 5 holds measured numbers.

**Milestone 1B — Viewer**

- App shell, worker bridge, trace reader.
- Graph rendering on React Flow with automatic layout (dagre).
- Stepper: play, pause, step forward, step back, scrub; forward pass then backward pass.
- Node and edge inspectors: value, gradient, shape, local derivative, the chain-rule product for that step.
- Scalar and tensor zoom levels.
- Reverse and forward mode side by side on the same graph, with pass counts.
- Backward ready queue and gradient accumulation shown as steps.
- Presets: a single expression, one neuron, a 2-layer MLP, a broadcasting example.

Done when: a user can step through a 2-layer MLP and read every intermediate value and gradient; forward and reverse mode show matching gradients on the presets; stepping stays responsive on the largest preset.

**Milestone 1C — Live training**

- Losses: MSE, cross-entropy with softmax.
- Optimizers: SGD, momentum, Nesterov, RMSProp, Adam, AdamW; schedules; gradient clipping.
- Initialisers: uniform, normal, Xavier, He.
- Datasets and batching (2D toy generators ported from TensorFlow Playground's approach).
- Training controls: run, pause, single step, reset, speed.
- Views: decision boundary, loss curve, per-layer activation and gradient histograms, gradient norm per layer over time, optimizer state buffers.
- Optimizer race: several optimizers on one 2D loss surface.
- Two configurations of one preset trained side by side.
- Presets that reproduce vanishing gradients and dead ReLUs.
- Landing page and first public release.

Done when: the same network trains with two optimizers side by side from one seed; the pathology presets show the effect in the charts; live training holds the frame budget at summary trace level; the site is deployed.

### Phase 2 — Builder: make your own, and launch

Goal: build a network from an empty canvas, cover the remaining mechanics, and launch.

**Milestone 2A — Builder**

- Node palette generated from the op registry.
- Drag, drop, connect, delete, undo and redo.
- Live shape inference; errors shown on the offending edge.
- Module grouping and collapse (Linear, MLP block, residual block).
- Skip connections, concat, weight sharing.
- Dropout and weight decay.
- Resize layers and change activations while paused, then continue.
- Save locally; share by URL.

Done when: a new user builds a residual MLP from an empty canvas and trains it without reading documentation; an invalid connection is explained where it occurs; a shared link reproduces the same graph and run.

**Milestone 2B — Depth**

- Convolution and pooling with kernel and feature-map views.
- Batch norm and layer norm.
- Memory timeline of saved tensors; checkpoint regions with visible recomputation.
- Single-head attention block (stretch).
- PyTorch code export of the built graph.

Done when: checkpointing shows lower peak memory and extra compute on the same network; exported PyTorch code reproduces the engine's gradients for the gallery examples.

**Milestone 2C — Launch**

- Example gallery and first-run onboarding.
- Performance pass, accessibility pass, browser coverage.
- Launch materials.

Done when: the launch checklist is complete.

### Why this order

The builder is an editor for a graph format that the engine, viewer and training loop must already execute and display, so it comes second. Phase 1 on its own is a complete product (a steppable, trainable set of presets), which gives a public release and real feedback before the editor is built.

## 8. The end product

One page, desktop-first, nothing to install and no account.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Examples ▾   Share   Export PyTorch         View: Scalar | Tensor | Module│
├───────────┬───────────────────────────────────────────────┬───────────────┤
│ PALETTE   │                    CANVAS                     │ INSPECTOR     │
│ Data      │                                               │ selected node │
│ Linear    │  [x]─▶[Linear 2→8]─▶[ReLU]─▶[Linear 8→8]─┐    │  value heatmap│
│ Conv      │           │                              ▼    │  grad heatmap │
│ Activation│           └──────── skip ─────────────▶[Add]  │  shape, op    │
│ Norm      │                                        │      │  local deriv. │
│ Add/Concat│                              [Linear 8→2]     │  histogram    │
│ Loss      │                                   ▼           │  optimizer    │
│ Optimizer │                            [CrossEntropy]     │   state       │
├───────────┴───────────────────────────────────────────────┴───────────────┤
│ ⏮ ◀ ▶ ⏭  Step ▸ Run ▸ Reset   speed ━━●━━   pass: forward | backward      │
│ timeline: [forward ops ··········][backward ops ··········][update]       │
├───────────────────────────────────────────────────────────────────────────┤
│ Decision boundary │ Loss curve │ Grad norm per layer │ Memory │ Optimizers│
└───────────────────────────────────────────────────────────────────────────┘
```

What a session looks like:

1. **Build.** Drag a data source, layers, an activation, a loss and an optimizer onto the canvas and connect them. Shapes appear on every edge as you connect; a mismatch turns the edge red and says which dimensions disagree. Drag a second wire around a block to make a skip connection.
2. **Step.** Press step. One op lights up, its output tensor fills in as a heatmap, and the timeline advances by one event. At the loss, the direction reverses: gradients flow back edge by edge, and at a node with two consumers the two incoming gradients are shown being added. Scrub the timeline to go back.
3. **Zoom.** Switch a small layer to scalar view and it unfolds into individual numbers and multiplications, as in micrograd. Switch to module view and a Linear layer collapses to one box.
4. **Train.** Press run. The decision boundary bends, the loss curve falls, histograms of each layer's activations and gradients update, and edge thickness tracks gradient size so a vanishing gradient is something you see fade.
5. **Compare.** Duplicate the run with a different optimizer, activation or initialisation and watch both from the same seed. Open the optimizer panel to see the momentum and second-moment buffers behind each update.
6. **Look under the hood.** Switch to forward mode and see one pass per input instead of one pass per output. Open the memory panel to see which activations are held for backward; draw a checkpoint region and watch peak memory fall while that segment recomputes.
7. **Take it away.** Share a link that reopens the exact network and run, or export the canvas as PyTorch code.

Phase 1 delivers steps 2 to 6 on preset networks, without the palette, checkpoint regions or export. Phase 2 adds step 1, step 7, convolution, normalisation and the memory tools.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Reporting to the page makes steps too slow | Statistics computed in the engine; full tensors only for the selected node; one frame in flight; measure in milestone 1A |
| Large graphs become unreadable | Module collapse, zoom levels, node cap |
| Engine gradients are subtly wrong | Finite-difference and PyTorch comparison on every op, in CI |
| Scope growth toward "all of deep learning" | The syllabus table is the scope; additions go through the PRD |
| Scalar and tensor views drift apart | Scalar view is derived from the tensor trace, never a second engine |

## 10. Open decisions

| Decision | Default | Decide by |
|---|---|---|
| Frontend stack | TypeScript, React, React Flow, dagre | Start of milestone 1B |
| Product name | None yet | Before the Phase 1 release |
| Accounts and saved work on a server | No; URL and local storage only | Milestone 2A |
| WebGPU backend | Not in this plan | After launch |
| Open source licence | Undecided; must be compatible with MIT and Apache-2.0 code we port | Before the Phase 1 release |
