"""Compare the engine against PyTorch.

Reads the JSON written by `ag_test_engine --dump FILE`: every op case (graph,
leaf values, every node's value and gradient as the engine computed them) and
ten steps of every optimizer. Rebuilds each in torch, float32, and compares.

    python tests/torch_check.py build/cases.json

Exits non-zero on any mismatch.
"""

import json
import sys

import torch
import torch.nn.functional as F

ALL_AXES = -128


def build(nodes):
    """Evaluate the dumped graph in torch. Leaves take the engine's values."""
    out = []
    for n in nodes:
        op, ins = n["op"], [out[i] for i in n["inputs"]]
        shape = n["shape"]
        if op in ("input", "param", "const"):
            t = torch.tensor(n["value"], dtype=torch.float32).reshape(shape)
            t.requires_grad_(len(n["grad"]) > 0)
        elif op in ("add", "sub", "mul", "div"):
            t = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[op](ins[0], ins[1])
        elif op == "neg":
            t = -ins[0]
        elif op == "pow":
            t = ins[0] ** n["scalar"]
        elif op in ("exp", "log", "tanh", "sigmoid", "relu"):
            t = getattr(torch, op)(ins[0])
        elif op == "leaky_relu":
            t = F.leaky_relu(ins[0], n["scalar"])
        elif op == "matmul":
            t = ins[0] @ ins[1]
        elif op in ("sum", "mean"):
            x = ins[0]
            if n["axis"] == ALL_AXES:
                dims = tuple(range(x.dim()))
                t = getattr(x, op)(dim=dims, keepdim=True) if n["keepDim"] else getattr(x, op)()
            else:
                t = getattr(x, op)(dim=n["axis"], keepdim=n["keepDim"])
        elif op == "reshape":
            t = ins[0].reshape(shape)
        elif op == "transpose":
            t = ins[0].t()
        elif op == "softmax":
            t = torch.softmax(ins[0], dim=-1 if n["axis"] == ALL_AXES else n["axis"])
        elif op == "mse_loss":
            t = F.mse_loss(ins[0], ins[1])
        elif op == "softmax_cross_entropy":
            z, y = ins
            rows = z.numel() // z.shape[-1]
            t = -(y * torch.log_softmax(z, dim=-1)).sum() / rows
        else:
            raise ValueError("no torch rule for " + op)
        if t.requires_grad and not t.is_leaf:
            t.retain_grad()
        out.append(t)
    return out


def close(a, b, tol):
    a = torch.as_tensor(a, dtype=torch.float32).flatten()
    b = torch.as_tensor(b, dtype=torch.float32).flatten()
    if a.shape != b.shape:
        return False, float("inf")
    err = ((a - b).abs() / torch.clamp(torch.maximum(a.abs(), b.abs()), min=1.0)).max().item() if a.numel() else 0.0
    return err <= tol, err


def check_case(case):
    nodes = case["nodes"]
    ts = build(nodes)
    ts[case["loss"]].backward()
    worst = 0.0
    for i, n in enumerate(nodes):
        ok, err = close(ts[i].detach(), n["value"], 1e-5)
        worst = max(worst, err)
        if not ok:
            return False, f"value of node {i} ({n['op']}) differs, rel err {err:.2e}"
        if n["grad"]:
            ok, err = close(ts[i].grad, n["grad"], 1e-4)
            worst = max(worst, err)
            if not ok:
                return False, f"grad of node {i} ({n['op']}) differs, rel err {err:.2e}"
    return True, f"worst rel err {worst:.1e}"


def check_optimizer(o):
    steps = o["steps"]
    p = torch.tensor(steps[0], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(o["t"], dtype=torch.float32)
    # the engine stores its settings as float32; give torch the same numbers.
    # 0.999 and float32(0.999) differ by 1.3e-8, and rmsprop's first steps
    # (about lr / sqrt(1 - alpha) in size) amplify that past the tolerance
    f32 = lambda x: torch.tensor(x, dtype=torch.float32).item()
    lr, wd = f32(o["lr"]), f32(o["weightDecay"])
    mom, b1, b2, eps = f32(0.9), f32(0.9), f32(0.999), f32(1e-8)
    make = {
        "sgd": lambda: torch.optim.SGD([p], lr=lr, weight_decay=wd),
        "momentum": lambda: torch.optim.SGD([p], lr=lr, momentum=mom, weight_decay=wd),
        "nesterov": lambda: torch.optim.SGD([p], lr=lr, momentum=mom, nesterov=True, weight_decay=wd),
        "rmsprop": lambda: torch.optim.RMSprop([p], lr=lr, alpha=b2, eps=eps, weight_decay=wd),
        "adam": lambda: torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
        "adamw": lambda: torch.optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd),
    }[o["kind"]]
    opt = make()
    worst = 0.0
    for k in range(1, len(steps)):
        opt.zero_grad()
        F.mse_loss(p ** 3, t).backward()
        torch.nn.utils.clip_grad_norm_([p], o["clipNorm"])
        opt.step()
        ok, err = close(p.detach(), steps[k], 1e-4)
        worst = max(worst, err)
        if not ok:
            return False, f"step {k} differs, rel err {err:.2e}"
    return True, f"10 steps, worst rel err {worst:.1e}"


def main():
    data = json.load(open(sys.argv[1]))
    failed = 0
    print("[ops against torch]")
    for case in data["cases"]:
        ok, msg = check_case(case)
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {case['name']}: {msg}")
    print("[optimizers against torch.optim]")
    for o in data["optim"]:
        ok, msg = check_optimizer(o)
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {o['kind']}: {msg}")
    print(f"=== {'all passed' if not failed else f'{failed} failed'} ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
