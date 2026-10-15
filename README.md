# gpu-puzzles-triton

[Sasha Rush's GPU Puzzles](https://github.com/srush/GPU-Puzzles) re-implemented with
**Triton** kernels.

## Why this exists

The upstream notebook is excellent teaching material, but its grader drives `numba.cuda`,
which needs **libNVVM** — a library that ships only with a full CUDA Toolkit. On a machine
without the Toolkit, every kernel dies at the first call with:

```
NvvmSupportError: libNVVM cannot be found. Do `conda install cudatoolkit`:
Could not find module 'nvvm.dll' (or one of its dependencies).
```

That is a common situation: you have a GPU, a CUDA build of PyTorch and Triton, but never
installed the several-GB Toolkit (and `numba` 0.61+ moved CUDA support into a separate
`numba-cuda` package on top of that).

Triton ships its own compiler (LLVM + `ptxas`) and needs no Toolkit, so these versions run
wherever `torch` + `triton` do — including a ComfyUI or Stable Diffusion WebUI embedded
Python, which is often the only CUDA-ready interpreter on a Windows box.

## What is here

`triton_puzzles.py` — eight exercises in the original order, each solved as a Triton kernel
and graded against the original numpy specification:

| # | Puzzle | Kernel idea it teaches |
|---|---|---|
| 1 | Map | one program per tile; `tl.load` / `tl.store` |
| 2 | Zip | two inputs, one output |
| 3 | Guards | `tl.where` so out-of-range lanes still write their sentinel |
| 4 | Map 2D | flat indexing across a matrix |
| 5 | Broadcast | a per-row scalar plus a per-column vector |
| 6 | Blocks | `tl.program_id` to cover a large array |
| 10 | Dot product | accumulate inside a block, reduce with `tl.sum` |
| — | Softmax | row-wise max / exp / sum, the shape every attention kernel uses |

Every kernel is checked against a numpy reference implementation with `np.allclose`, so a
wrong result fails loudly instead of silently.

## Run it

```bash
python triton_puzzles.py
```

It prints `PASS`/`FAIL` per puzzle plus a summary, and exits non-zero if anything failed.
On an RTX 4090 the whole file (including Triton's kernel compilation) finishes in about
2.4 seconds.

## Requirements

- Python with a CUDA build of `torch` and `triton`
- An NVIDIA GPU

## Credits and scope

The exercises, their intent and their specifications are Sasha Rush's
([srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles), MIT). This file only re-expresses
them as Triton kernels with a numpy grader, so that they can be attempted without a CUDA
Toolkit. If upstream ever grows a Triton track, that one should win.

## License

MIT — see [LICENSE](LICENSE).
