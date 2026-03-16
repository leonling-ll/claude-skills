---
name: triton-to-gluon
description: >
  Translate a Triton GPU kernel to a Gluon kernel for AMD GPUs (MI300X / MI308X / MI350).
  Workflow: (1) detect GPU architecture (gfx942→cdna3, gfx950→cdna4), (2) run the Triton
  kernel to dump ttgir/llir from the Triton cache, (3) parse the IR to extract precise
  tensor layouts and pipeline structure, (4) produce a Gluon kernel that explicitly sets
  BlockedLayout / SwizzledSharedLayout / AMDMFMALayout / DotOperandLayout for every tensor,
  (5) add verify_correctness() and profile_kernel() using torch.profiler, (6) fix bugs,
  (7) save the output file and clean up the cache.
  Use this skill whenever the user asks to: convert a Triton kernel to Gluon, rewrite a
  @triton.jit kernel using gluon.jit, translate a triton kernel for AMD Gluon, or port a
  Triton matmul/attention/linear kernel to use explicit AMD layouts and MFMA instructions.
  Usage: /triton-to-gluon <kernel_file.py>
tools: Read,Edit,Bash,Grep,Glob,Write,Agent
---

# Triton → Gluon Translation

Translate a `@triton.jit` kernel into a `@gluon.jit` kernel that uses explicit AMD tensor
layouts and MFMA instructions.  The IR files (ttgir / llir) are the ground truth for every
layout decision — read them before writing a single line of Gluon code.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<kernel_file.py>` | Yes | Path to the Python file containing the Triton kernel |

---

## Step 1 — Detect GPU architecture

```bash
python -c "
import subprocess, re
out = subprocess.check_output(['rocminfo'], text=True)
gfx = re.findall(r'gfx\d+', out)
print(next((g for g in gfx if g.startswith('gfx9')), 'unknown'))
"
```

| gfx ID | Architecture | Gluon namespace | XCDs |
|--------|-------------|-----------------|------|
| gfx942 | CDNA3 (MI300X / MI308X) | `gl.amd.cdna3` | 8 (MI300X) / 4 (MI308X) |
| gfx950 | CDNA4 (MI350) | `gl.amd.cdna4` | TBD |

Determine XCD count from CU count reported by `torch.cuda.get_device_properties(0)`:
- 304 CUs → MI300X (8 XCDs)
- ~152 CUs → MI308X (4 XCDs)

---

## Step 2 — Dump ttgir and llir

Derive `<kernel_name>` from the Python filename (without `.py`).

```bash
TRITON_CACHE_DIR="./triton_cache/<kernel_name>/" python <kernel_file.py>
```

If the kernel file has a `__main__` guard that runs a benchmark or profiler, that is fine —
the compilation artifacts are written on the first kernel call regardless.

---

## Step 3 — Locate the IR files

```bash
find ./triton_cache/<kernel_name>/ -name "*.ttgir" -o -name "*.llir" | sort
```

Pick the `.ttgir` file whose embedded function name matches the `@triton.jit` function name
in the source file.  Also pick the corresponding `.llir` (same hash prefix).

Read **both** files in full before proceeding.  The ttgir shows you the high-level layout
annotations; the llir shows you the exact instruction order and any low-level ops.

---

## Step 4 — Extract layout information from ttgir

Every tensor type in ttgir carries an encoding attribute.  Map each one to the Gluon layout:

| ttgir encoding | Gluon layout |
|----------------|--------------|
| `#blocked` on A (input) | `gl.BlockedLayout(size_per_thread=[1, K], threads_per_warp=[M_t, K_t], warps_per_cta=[W, 1], order=[1,0])` |
| `#blocked` on B (weight) | `gl.BlockedLayout(size_per_thread=[K, 1], threads_per_warp=[K_t, N_t], warps_per_cta=[1, W], order=[0,1])` |
| `#shared` on smem A | `gl.SwizzledSharedLayout(vec=V, per_phase=P, max_phase=MP, order=[1,0])` |
| `#shared1` on smem B | `gl.SwizzledSharedLayout(vec=V, per_phase=P, max_phase=MP, order=[0,1])` |
| `#mma` / `#amdmfma` | `gl.amd.AMDMFMALayout(version=3, instr_shape=[M,N,K], transposed=True, warps_per_cta=[1, W])` |
| dot operand 0 | `gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=KW)` |
| dot operand 1 | `gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=KW)` |

`k_width` = `size_per_thread` in the K dimension of the corresponding blocked layout.

Typical values for bf16 with `matrix_instr_nonkdim=16`, `num_warps=4`, `BLOCK_K=64`:
- `size_per_thread = [1, 16]` for A → `k_width = 16`
- blocked A: `threads_per_warp=[16,4], warps_per_cta=[4,1]`
- blocked B: `size_per_thread=[16,1], threads_per_warp=[4,16], warps_per_cta=[1,4]`
- mfma: `instr_shape=[16,16,16], warps_per_cta=[1,4]`
- shared: `vec=8, per_phase=2, max_phase=4`

Always verify against the actual ttgir — these are examples, not universal constants.

---

## Step 5 — Write the Gluon kernel

### Imports

```python
import torch, triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

### XCD remapping helper (paste as-is, adjust NUM_XCDS)

```python
@triton.jit
def remap_xcd(pid, grid_mn, NUM_XCDS: tl.constexpr):
    pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = grid_mn % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (tall_xcds * pids_per_xcd
               + (xcd - tall_xcds) * (pids_per_xcd - 1)
               + local_pid)
    return pid
```

### Kernel signature

- Replace `tl.constexpr` with `gl.constexpr`
- Replace `@triton.jit` with `@gluon.jit`
- Keep `@triton.autotune` as-is (it still works with `@gluon.jit`)
- Add any batch/shape constants needed for XCD remapping (e.g., `B: gl.constexpr`)

### Kernel body — translation map

| Triton | Gluon |
|--------|-------|
| `tl.program_id(0)` | `gl.program_id(axis=0)` |
| `tl.cdiv(a, b)` | `gl.cdiv(a, b)` |
| `tl.arange(0, N)` | `gl.arange(0, N, layout=gl.SliceLayout(dim, parent_layout))` |
| `tl.zeros((M,N), dtype)` | `gl.zeros((M,N), dtype=gl.float32, layout=mfma_layout)` |
| `tl.load(ptr, mask=m)` | `gl.amd.cdna3.buffer_load(ptr=ptr, offsets=offs, mask=m)` |
| `tl.store(ptr, val, mask=m)` | `gl.amd.cdna3.buffer_store(stored_value=val, ptr=ptr, offsets=offs, mask=m)` |
| `tl.dot(a, b, acc)` | *(see pipeline section below)* |

**`gl.arange` and `gl.SliceLayout`:**
```python
# A is [BLOCK_M, BLOCK_K]; K dimension is axis=0 in blocked_mk SliceLayout
offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_mk))
offs_am = row_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
```
`SliceLayout(dim, parent)` picks the row (`dim=1`) or column (`dim=0`) slice of the parent
blocked layout.

### Explicit pipeline (replaces `num_stages` in Triton)

Triton's `num_stages=2` becomes a manual single-buffer pipeline in Gluon:

```python
# Allocate shared memory
smem_a = gl.allocate_shared_memory(x_ptr.type.element_ty,  [BLOCK_M, BLOCK_K], layout=shared_a)
smem_b = gl.allocate_shared_memory(w_ptr.type.element_ty,  [BLOCK_K, BLOCK_N], layout=shared_b)

# PROLOGUE: load tile 0 → smem
a0 = gl.amd.cdna3.buffer_load(ptr=x_ptr,   offsets=offs_a, mask=mask_a)
b0 = gl.amd.cdna3.buffer_load(ptr=w_ptr,   offsets=offs_b, mask=mask_b)
smem_a.store(a0);  smem_b.store(b0)

acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

# MAIN LOOP: tiles 1 … N-2
for k in range(0, gl.cdiv(K, BLOCK_K) - 1):
    cur_a = smem_a.load(layout=dot_a_layout)   # LDS read
    x_ptr += BLOCK_K * x_stride_k              # advance pointer
    next_a = gl.amd.cdna3.buffer_load(...)     # global load next tile
    cur_b = smem_b.load(layout=dot_b_layout)
    w_ptr += BLOCK_K * w_stride_k
    next_b = gl.amd.cdna3.buffer_load(...)
    acc = gl.amd.cdna3.mfma(cur_a, cur_b, acc)
    smem_a.store(next_a);  smem_b.store(next_b)

# EPILOGUE: last tile
cur_a = smem_a.load(layout=dot_a_layout)
cur_b = smem_b.load(layout=dot_b_layout)
acc   = gl.amd.cdna3.mfma(cur_a, cur_b, acc)
```

Verify this matches the ttgir pipeline order exactly (look at the sequence of
`local_load`, `amdg.buffer_load`, `tt.dot` / `amdg.mfma` in the IR).

### Unsupported IR instructions

Skip these with a comment — they are scheduler hints not exposed in Gluon:
- `rocdl.s.setprio`
- `rocdl.sched.barrier`
- `rocdl.waitcnt`

### Output store

```python
c = acc.to(output_ptr.type.element_ty)
offs_cm = row_start + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
offs_cn = pid_n*BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
c_offs  = pid_b*out_stride_b + offs_cm[:,None]*out_stride_m + offs_cn[None,:]*out_stride_n
c_mask  = (offs_cm[:,None] < M) & (offs_cn[None,:] < N)
gl.amd.cdna3.buffer_store(stored_value=c, ptr=output_ptr, offsets=c_offs, mask=c_mask)
```

Drop unnecessary mask dimensions when `EVEN_M` or `EVEN_N` are True (matching ttgir opt).

---

## Step 6 — Write the launcher

```python
def kernel_launcher(x, weight, bias):
    B, M, K = x.shape
    N = weight.shape[0]
    output = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (B * triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    # Gluon kernels require wrap_triton; bare kernel[grid](...) does not work
    torch.library.wrap_triton(matmul_kernel)[grid](
        x, weight, bias, output,
        B, M, K, N,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(1), weight.stride(0),
        bias.stride(0),
        output.stride(0), output.stride(1), output.stride(2),
        EVEN_K=(K % BLOCK_K == 0),
        EVEN_N=(N % 128 == 0),
        EVEN_M=(M % 128 == 0),
    )
    return output
```

---

## Step 7 — Add verify_correctness and profile_kernel

```python
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

def verify_correctness():
    torch.manual_seed(0)
    # Use small shapes that exercise boundary conditions (non-power-of-2 M)
    B, M, K, N = 4, 100, 256, 128
    x      = torch.randn(B, M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    bias   = torch.randn(N,    dtype=torch.bfloat16, device="cuda")
    out_g  = kernel_launcher(x, weight, bias)
    out_r  = F.linear(x, weight, bias)
    err    = (out_g - out_r).abs().max().item()
    assert err < 1.0, f"max error {err:.4f} exceeds tolerance"
    print(f"Correctness passed. Max error: {err:.4f}")

def profile_kernel(B=80, M=248, K=3072, N=768, warmup=10, steps=50):
    x      = torch.randn(B, M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    bias   = torch.randn(N,    dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        kernel_launcher(x, weight, bias)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(steps):
            kernel_launcher(x, weight, bias)
    torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace_gluon.json")

if __name__ == "__main__":
    verify_correctness()
    profile_kernel()
```

---

## Step 8 — Fix bugs and save

Run the new file and fix any errors:

```bash
python <kernel_file>_gluon.py
```

Common issues and fixes:

| Error | Cause | Fix |
|-------|-------|-----|
| `k_width` mismatch | `DotOperandLayout.k_width` ≠ `size_per_thread[K dim]` | Recompute: `k_width = BLOCK_K // threads_per_warp_k` |
| Shape mismatch in `mfma` | Wrong layout on `cur_a` or `cur_b` | Confirm `dot_a_layout` uses `operand_index=0`, `dot_b_layout` uses `operand_index=1` |
| Out-of-bounds store | `c_mask` too permissive | Add both row (`< M`) and column (`< N`) guards |
| Wrong output values | Pipeline order wrong | Re-read ttgir and match the exact `local_load / buffer_load / mfma` sequence |
| `wrap_triton` error | Kernel launched without wrapper | Use `torch.library.wrap_triton(kernel)[grid](...)` |

Once `verify_correctness()` passes, save the file next to the original:

```
<original_dir>/<kernel_name>_gluon.py
```

Then clean up the cache:

```bash
rm -rf ./triton_cache/<kernel_name>/
```
