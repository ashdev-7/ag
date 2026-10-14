# Product Requirements Document

A browser tool that shows what a neural network is doing while it trains, down to individual numbers. The build sequence is in [PLAN.md](PLAN.md).

Status: draft, revised 2026-10-14 (two phases; prior-art review added). Product name: not chosen.

## 1. Problem

People learning deep learning are taught backpropagation, optimizers, activations and architectures mostly through equations and through code whose internals are hidden. In PyTorch, `loss.backward()` is one line; what it does to each tensor is invisible. Learners end up able to run training without being able to say what happened inside it.

## 2. Product in one sentence

A transparent pane over a small autograd engine: build a network by dragging and connecting nodes, then step through or watch the forward pass, the backward pass and the parameter updates on every tensor.

## 3. Who it is for

| User | What they want |
|---|---|
| Student learning backpropagation | To see the chain rule applied step by step on a real graph |
| Self-taught practitioner who uses PyTorch | To understand what autograd, optimizers and checkpointing do underneath |
| Instructor | A live demonstration to project, and links to hand to students |
| Interview candidate | To check their mental model of autograd mechanics |

Not the target: people training production models, or researchers who need scale.

## 4. Existing tools and the gap

| Tool | What it covers | What it does not |
|---|---|---|
| TensorFlow Playground | Dense networks on 2D toy data, live training | Fixed architecture family; no autograd internals |
| ENNUI | Drag-and-drop builder, in-browser training, Python export | Layer level only, on TensorFlow.js; no view inside backward; last updated 2023 |
| CNN Explainer, Transformer Explainer, llm-viz | One fixed model each, forward pass in detail | No training, no editing |
| Netron | Structure of saved models | No execution |
| autograd-ui | Scalar autograd graph in the browser | Scalars only; no tensors, training or builder |
| Optimizer playgrounds (several) | Optimizer paths on fixed 2D surfaces | Not connected to a network the user built |

Each piece of this product exists somewhere. What was not found is one tool that joins them: build an arbitrary small network, then step through autograd on it op by op, in both modes, and watch it train. Positioning should therefore be "the pieces, joined, on a transparent engine", not "nothing like this exists".

Licences and activity were checked on 2026-10-14; the full table and what we reuse from each is in PLAN.md section 3.2. The search was not exhaustive.

## 5. Goals

1. Every number involved in training is inspectable: values, gradients, saved tensors, optimizer state.
2. Training is something the user watches and controls: run, pause, step, rewind within a step.
3. Users build their own networks rather than choosing from fixed ones.
4. It runs entirely in the browser with no install and no account.
5. Results are correct: gradients match PyTorch on the same graph.

## 6. Non-goals

- Training models of practical size. Limits are in section 10.
- Being a deep learning course. The first versions carry little explanatory text; the visualisations are the content.
- GPU execution.
- Importing arbitrary PyTorch or ONNX models.
- Collaboration, accounts or server-side storage.
- Mobile-first design. The canvas targets desktop screens.

## 7. User stories

Viewer:
- As a student, I load a small expression and step through the backward pass, seeing which local derivative multiplies which incoming gradient.
- As a student, I see why a value used twice receives the sum of two gradients.
- As a learner, I run the same graph in forward mode and reverse mode and see how many passes each needs.
- As a learner, I switch between a per-scalar view and a per-tensor view of the same computation.

Training:
- As a learner, I watch a decision boundary form while the loss falls.
- As a learner, I train one network with SGD and with Adam side by side and compare paths and state.
- As a learner, I make a network deep with sigmoid activations and watch early-layer gradients shrink.
- As a learner, I change the initialisation and see the activation distributions change before any training.

Builder:
- As a user, I drag layers onto a canvas, connect them and get told at the edge when shapes do not match.
- As a user, I add a skip connection and compare gradient flow with and without it.
- As a user, I change a layer's width or activation and continue training.
- As an instructor, I share a link that opens my exact network and run.

Advanced:
- As a practitioner, I mark part of a network as checkpointed and see peak memory fall and compute rise.
- As a practitioner, I export my canvas as PyTorch code.

## 8. Functional requirements

Priority: P0 is required for that milestone, P1 is wanted, P2 is optional.

### 8.1 Engine (Phase 1, milestone 1A)

| ID | Requirement | Priority |
|---|---|---|
| E1 | N-dimensional float32 tensors with shape and broadcasting | P0 |
| E2 | Each op defines forward, reverse-mode and forward-mode derivative, and shape inference | P0 |
| E3 | Reverse-mode and forward-mode execution over the same graph | P0 |
| E4 | A trace of forward, backward, save, release and update events at selectable detail | P0 |
| E5 | Graph description separate from execution, serialisable | P0 |
| E6 | Runs as WebAssembly in a Web Worker | P0 |
| E7 | Deterministic given a seed | P0 |
| E8 | Gradients verified against finite differences and PyTorch for every op | P0 |
| E9 | No memory growth across repeated training steps | P0 |
| E10 | `requires_grad`, `no_grad`, `detach` | P1 |

### 8.2 Viewer (Phase 1, milestone 1B)

| ID | Requirement | Priority |
|---|---|---|
| V1 | Render a graph with automatic layout | P0 |
| V2 | Play, pause, step forward, step back and scrub through one forward and backward pass | P0 |
| V3 | Inspect any node or edge: value, gradient, shape, local derivative | P0 |
| V4 | Scalar and tensor zoom levels | P0 |
| V5 | Reverse and forward mode side by side with pass counts | P0 |
| V6 | Show gradient accumulation at nodes with several consumers | P0 |
| V7 | Preset examples | P0 |
| V8 | Show the backward execution order and ready queue | P1 |

### 8.3 Training (Phase 1, milestone 1C)

| ID | Requirement | Priority |
|---|---|---|
| T1 | Losses: MSE, cross-entropy | P0 |
| T2 | Optimizers: SGD, momentum, Adam | P0 |
| T3 | Optimizers: Nesterov, RMSProp, AdamW; learning-rate schedules; gradient clipping | P1 |
| T4 | Initialisers: Xavier, He, plain uniform and normal | P0 |
| T5 | Built-in small datasets with batching | P0 |
| T6 | Training controls: run, pause, single step, reset, speed | P0 |
| T7 | Decision boundary and loss curve | P0 |
| T8 | Per-layer activation and gradient histograms; gradient norm per layer over time | P0 |
| T9 | Optimizer state shown per parameter | P1 |
| T10 | Several optimizers compared on one 2D loss surface | P0 |
| T11 | Two configurations of one network trained side by side | P1 |

### 8.4 Builder (Phase 2, milestone 2A)

| ID | Requirement | Priority |
|---|---|---|
| B1 | Palette of nodes; drag, drop, connect, delete | P0 |
| B2 | Undo and redo | P0 |
| B3 | Live shape inference with errors shown at the offending edge | P0 |
| B4 | Group nodes into collapsible modules | P0 |
| B5 | Skip connections and concatenation | P0 |
| B6 | Edit layer size and activation while paused, then continue | P0 |
| B7 | Save locally; share by URL | P0 |
| B8 | Weight sharing between nodes | P1 |
| B9 | Dropout and weight decay | P1 |

### 8.5 Depth and launch (Phase 2, milestones 2B and 2C)

| ID | Requirement | Priority |
|---|---|---|
| A1 | Memory timeline of saved tensors across a step | P0 |
| A2 | Checkpoint regions with visible recomputation in backward | P0 |
| A3 | Convolution and pooling with kernel and feature-map views | P0 |
| A4 | Batch norm and layer norm | P1 |
| A5 | PyTorch code export | P1 |
| A6 | Example gallery and first-run onboarding | P0 |
| A7 | Single-head attention block | P2 |

## 9. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | A training step completes within one frame (16 ms) at summary trace level, within the limits in section 10 |
| Responsiveness | The interface never blocks on compute; all engine work is in a worker |
| Correctness | Gradients match PyTorch within float32 tolerance on the test suite |
| Reproducibility | A shared link reproduces the same graph and the same run |
| Delivery | Static site; no backend needed to use the product |
| Browsers | Current desktop Chrome, Edge, Firefox and Safari |
| Privacy | No account; nothing leaves the browser except what the user puts in a shared link |
| Accessibility | Keyboard control of the stepper; colour scales that do not rely on red against green |

## 10. Limits

These are targets to be confirmed by measurement in milestone 1A.

| Limit | Target |
|---|---|
| Parameters per model | about 100k |
| Elements per tensor | about 64k |
| Nodes on canvas | about 200 at tensor level |
| Scalar view | tensors up to about 500 scalars |
| Data | Bundled small datasets only |

When a user exceeds a limit, the product says which limit and why it exists, instead of slowing down or failing silently.

## 11. Success measures

Proposed measures; targets to be set once the Phase 1 release gives a baseline.

| Measure | What it tells us |
|---|---|
| Share of visitors who step through at least one full backward pass | Whether the viewer is understood |
| Share of visitors who start a training run | Whether people get past stepping to training |
| Share of builder sessions that produce a trainable network | Whether the editor is usable unaided |
| Shared links created | Whether instructors and learners pass it on |
| Return visits | Whether it is a reference, not a one-off demo |

Collecting any of these needs analytics, which conflicts with "nothing leaves the browser". This is an open question in section 13.

## 12. Release milestones

| Release | After | Contents | Audience |
|---|---|---|---|
| Internal | Milestone 1A | Engine, tests, benchmarks | Developer only |
| Internal preview | Milestone 1B | Read-only stepper on presets | A few invited learners |
| Phase 1 public release | Milestone 1C | Stepper plus live training and comparisons on presets | Public, for feedback |
| Internal preview | Milestone 2A | Builder and shared links | Invited instructors and learners |
| Launch | Milestone 2C | Builder, convolution, normalisation, memory tools, export, gallery, onboarding | Product Hunt |

## 13. Open questions

1. Product name and domain.
2. Open source or closed, and which licence.
3. Whether to add privacy-preserving analytics, given section 11.
4. Whether any explanatory text ships with presets, or strictly none at first.
5. Whether a paid tier is ever intended; this affects whether accounts are designed for.
6. Whether mobile viewing (not building) is worth supporting.

## 14. Risks

| Risk | Effect | Response |
|---|---|---|
| Visualisations are impressive but do not teach | Low retention | Release the viewer first and test with learners before building the editor |
| Limits feel too small | Users dismiss it as a toy | Explain the limits; choose examples where small networks show the real effect |
| Canvas becomes cluttered | Unusable beyond trivial graphs | Module collapse and zoom levels from the start |
| Scope grows without end | Never launches | The syllabus table in PLAN.md is the scope; changes are recorded here |
| A wrong gradient is shown | Loss of trust in a teaching tool | Automated comparison against PyTorch on every op |
