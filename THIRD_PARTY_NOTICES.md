# Third-party notices

## Code this project depends on

The engine (`engine/`) contains no third-party code.

The site uses these libraries, installed from npm and bundled into the build:

| Package | Licence | Use |
|---|---|---|
| react, react-dom | MIT | UI |
| @xyflow/react (React Flow) | MIT | graph canvas |
| @dagrejs/dagre | MIT | graph layout |

Development only, not shipped: vite, @vitejs/plugin-react, typescript, vitest (MIT); playwright (Apache-2.0).

The PCG32 generator in `engine/src/rng.cpp` follows Melissa O'Neill's published
algorithm (pcg-random.org); the code is our own.

## Projects that informed the design

No code was copied from these. Their ideas are credited here and in
`product-dev/DESIGN_PHASE1.md` §8.

| Project | Licence | What we learned from it |
|---|---|---|
| micrograd (Andrej Karpathy) | MIT | the shape of a minimal autodiff engine |
| TensorFlow Playground | Apache-2.0 | 2-d toy datasets, the decision-boundary grid, state in the URL |
| CNN Explainer | MIT | drawing from a complete snapshot of the model |
| Transformer Explainer | MIT | precomputed examples when the model cannot run |
| Netron | MIT | one error surface; a worker the page can terminate |
