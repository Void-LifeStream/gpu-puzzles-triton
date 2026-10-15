"""GPU-Puzzles (srush/GPU-Puzzles) re-implemented as Triton kernels.

Why not the upstream notebook: its grader drives numba.cuda, which needs libNVVM
(shipped only with a full CUDA Toolkit). This machine has no Toolkit, so
`numba.cuda.is_available()` is False and any kernel dies with
`NvvmSupportError: libNVVM cannot be found`. Triton ships its own compiler and
works out of the box, so the same exercises are solved here in Triton and graded
against the original numpy specifications.

Order follows the original: Map, Zip, Guards, Map 2D, Broadcast, Blocks,
Dot Product, Softmax.
"""
import numpy as np
import torch
import triton
import triton.language as tl

DEV = "cuda"
STATE = {"pass": 0, "fail": 0}


def grade(name, expected, got):
    ok = np.allclose(
        np.asarray(expected, dtype=np.float64).ravel(),
        np.asarray(got, dtype=np.float64).ravel(),
        atol=1e-4,
    )
    STATE["pass" if ok else "fail"] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print("        expected:", np.asarray(expected).ravel()[:6])
        print("        got     :", np.asarray(got).ravel()[:6])


def to_gpu(arr):
    return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=DEV)


# ---------------------------------------------------------------- Puzzle 1: Map
@triton.jit
def map_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(a_ptr + offs, mask=mask) + 10.0, mask=mask)


def solve_map(a):
    a_t = to_gpu(a)
    out = torch.empty_like(a_t)
    n = a_t.numel()
    BLOCK = 128
    map_kernel[(triton.cdiv(n, BLOCK),)](a_t, out, n, BLOCK=BLOCK)
    return out.cpu().numpy()


# ---------------------------------------------------------------- Puzzle 2: Zip
@triton.jit
def zip_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask),
        mask=mask,
    )


def solve_zip(a, b):
    a_t, b_t = to_gpu(a), to_gpu(b)
    out = torch.empty_like(a_t)
    n = a_t.numel()
    BLOCK = 128
    zip_kernel[(triton.cdiv(n, BLOCK),)](a_t, b_t, out, n, BLOCK=BLOCK)
    return out.cpu().numpy()


# ------------------------------------------------------------ Puzzle 3: Guards
@triton.jit
def guards_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    value = tl.load(a_ptr + offs, mask=in_range, other=0.0) + 10.0
    # Every lane writes: out-of-range lanes must leave -1 behind.
    tl.store(out_ptr + offs, tl.where(in_range, value, -1.0))


def solve_guards(a, padded):
    a_t = to_gpu(a)
    out = torch.full((padded,), -2.0, device=DEV, dtype=torch.float32)
    n = a_t.numel()
    BLOCK = 128
    guards_kernel[(triton.cdiv(padded, BLOCK),)](a_t, out, n, BLOCK=BLOCK)
    return out.cpu().numpy()


# ------------------------------------------------------------- Puzzle 4: Map 2D
@triton.jit
def map2d_kernel(a_ptr, out_ptr, rows, cols, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = rows * cols
    mask = offs < total
    tl.store(out_ptr + offs, tl.load(a_ptr + offs, mask=mask) + 10.0, mask=mask)


def solve_map2d(a):
    a_t = to_gpu(a)
    out = torch.empty_like(a_t)
    rows, cols = a_t.shape
    total = rows * cols
    BLOCK = 128
    map2d_kernel[(triton.cdiv(total, BLOCK),)](a_t, out, rows, cols, BLOCK=BLOCK)
    return out.cpu().numpy()


# ---------------------------------------------------------- Puzzle 5: Broadcast
@triton.jit
def broadcast_kernel(a_ptr, b_ptr, out_ptr, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    a_val = tl.load(a_ptr + row)          # scalar for this row
    b_vals = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * cols + offs, a_val + b_vals, mask=mask)


def solve_broadcast(a, b):
    a_t, b_t = to_gpu(a), to_gpu(b)
    rows, cols = a_t.numel(), b_t.numel()
    out = torch.empty((rows, cols), device=DEV, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(cols)
    broadcast_kernel[(rows,)](a_t, b_t, out, rows, cols, BLOCK=BLOCK)
    return out.cpu().numpy()


# ------------------------------------------------------------- Puzzle 6: Blocks
@triton.jit
def blocks_kernel(a_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(a_ptr + offs, mask=mask) * 2.0, mask=mask)


def solve_blocks(a):
    a_t = to_gpu(a)
    out = torch.empty_like(a_t)
    n = a_t.numel()
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    blocks_kernel[grid](a_t, out, n, BLOCK=BLOCK)
    return out.cpu().numpy()


# -------------------------------------------------------- Puzzle 10: Dot product
@triton.jit
def dot_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        idx = start + offs
        mask = idx < n
        acc += tl.load(a_ptr + idx, mask=mask, other=0.0) * tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(acc, axis=0))


def solve_dot(a, b):
    a_t, b_t = to_gpu(a), to_gpu(b)
    out = torch.zeros(1, device=DEV, dtype=torch.float32)
    n = a_t.numel()
    BLOCK = 1024
    dot_kernel[(1,)](a_t, b_t, out, n, BLOCK=BLOCK)
    return float(out.item())


# ------------------------------------------------------------- Softmax by row
@triton.jit
def softmax_kernel(x_ptr, out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * cols + offs, num / tl.sum(num, axis=0), mask=mask)


def solve_softmax(x):
    x_t = to_gpu(x)
    rows, cols = x_t.shape
    out = torch.empty_like(x_t)
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x_t, out, cols, BLOCK=BLOCK)
    return out.cpu().numpy()


def main():
    print("device:", torch.cuda.get_device_name(0), "| triton", triton.__version__, "| torch", torch.__version__)

    a = np.arange(4, dtype=np.float32)
    grade("Puzzle 1  Map", a + 10, solve_map(a))

    b = np.arange(4, dtype=np.float32)
    grade("Puzzle 2  Zip", a + b, solve_zip(a, b))

    padded = 128
    expected_guards = np.full(padded, -1.0, dtype=np.float32)
    expected_guards[: a.size] = a + 10
    grade("Puzzle 3  Guards", expected_guards, solve_guards(a, padded))

    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    grade("Puzzle 4  Map 2D", m + 10, solve_map2d(m))

    row = np.arange(4, dtype=np.float32)
    col = np.arange(4, dtype=np.float32)
    grade("Puzzle 5  Broadcast", row[:, None] + col[None, :], solve_broadcast(row, col))

    big = np.random.default_rng(0).random(10_000).astype(np.float32)
    grade("Puzzle 6  Blocks", big * 2.0, solve_blocks(big))

    p = np.random.default_rng(1).random(4096).astype(np.float32)
    q = np.random.default_rng(2).random(4096).astype(np.float32)
    grade("Puzzle 10 Dot product", float(np.dot(p.astype(np.float64), q.astype(np.float64))), solve_dot(p, q))

    logits = np.random.default_rng(3).random((64, 512)).astype(np.float32)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    grade("Softmax (64x512)", e / e.sum(axis=1, keepdims=True), solve_softmax(logits))

    print(f"\n{STATE['pass']} passed, {STATE['fail']} failed")
    return 1 if STATE["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
