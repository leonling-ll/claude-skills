---
name: gluon-mm-inst-opt
description: >
  Optimize memory access instructions in a Gluon kernel through three
  progressive steps: (1) replace gl.load/gl.store (flat pointer) with
  gl.amd.cdna3.buffer_load/buffer_store to eliminate masked-load branches and
  reduce address overhead; (2) on CDNA4 GPUs (gfx950/MI350), introduce LDS
  shared memory and replace buffer_load with gl.amd.cdna4.async_copy.buffer_load_to_shared
  (direct DMA from global to LDS), establishing the async pipeline structure
  needed for prefetch in later optimizations; (3) attention/multi-tile patterns —
  CDNA4 coalesced-write constraint, multi-buffer commit groups, double-buffered
  shared memory, permute-view tricks for transposed reuse, and async DMA for
  load-once operands like Q. Use when a Gluon kernel still uses gl.load/gl.store
  with mask= arguments OR when porting buffer_load patterns to async_copy in an
  attention/sparse-MLA kernel on gfx950.
  Usage: /gluon-mm-inst-opt
---

# Gluon GEMM: Memory Instruction Optimization

Optimize global memory access instructions in a Gluon GEMM kernel through two
progressive steps that match tutorial versions v1 and v2.

**Step A** (applies to all CDNA3/4 GPUs): Replace flat `gl.load`/`gl.store` with
AMD-native `buffer_load`/`buffer_store`.

**Step B** (CDNA4 / gfx950 only): Replace `buffer_load` with
`async_copy.buffer_load_to_shared` through LDS, establishing the DMA pipeline
that enables software-pipelined prefetch in later optimizations.

---

## Step 0: Check GPU Platform

```bash
python3 -c "import torch; props = torch.cuda.get_device_properties(0); print(props.name, props.gcnArchName)"
# OR
rocm-smi --showproductname 2>/dev/null
```

| GPU | Architecture | Step A (buffer_load) | Step B (async_copy) |
|-----|-------------|---------------------|---------------------|
| MI300X | gfx942 (CDNA3) | Yes | **No** |
| MI308X | gfx942 (CDNA3) | Yes | **No** |
| MI350  | gfx950 (CDNA4) | Yes | Yes |

On gfx942: apply Step A only, then proceed for LDS layout
improvements that do not require async_copy.

**MFMA instruction set note:**
- gfx942 (MI300X/MI308X): `version=3, instr_shape=[16, 16, 16]` in `AMDMFMALayout`
- gfx950 (MI350): `version=4, instr_shape=[16, 16, 32]`

---

## Step 1: Analyze the Current Kernel

Read the kernel and identify which memory path it currently uses:

```bash
grep -n "gl\.load\|gl\.store\|buffer_load\|async_copy\|allocate_shared_memory" <kernel_file>
```

| Finding | Action |
|---------|--------|
| `gl.load(ptr, mask=...)` present | Apply Step A |
| `buffer_load` present, no `async_copy` | Skip Step A, check platform for Step B |
| `async_copy` present | Both steps done — proceed to `/gluon-lds-opt` |

---

## Step A: Buffer Load/Store (All CDNA GPUs)

### Why

AMD CDNA GPUs have a dedicated `buffer_load_dwordx4` instruction that takes a
**scalar base pointer + per-thread 32-bit offset**. Compared with flat global loads:

- **Eliminates branches**: mask handling done in hardware via `v_cndmask` (no `s_cbranch`)
- **Reduces address register pressure**: scalar base instead of full vector of 64-bit pointers
- **Maps directly to hardware path**: bypasses flat-address TLB

Reference: v0 has ~140 branch instructions; v1 has 4.

### A.1 Change load: full pointer → base + offset

**Before (gl.load style):**
```python
a_ptrs = a_base + a_offsets      # full pointer tensor, updated every iteration
gl.load(a_ptrs, mask=mask, other=0.0)
# ... loop ...
a_ptrs += BLOCK_K * stride_ak    # update full 64-bit pointer tensor
```

**After (buffer_load style):**
```python
a_base = a_ptr + pid_m * BLOCK_M * stride_am   # scalar base only
a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak  # constant

gl.amd.cdna3.buffer_load(ptr=a_base, offsets=a_offsets, mask=mask, other=0.0)
# ... loop ...
a_base += BLOCK_K * stride_ak    # update scalar only (cheaper)
```

Apply the same pattern to the B tile.

### A.2 Change store

**Before:**
```python
c_ptrs = c_base + c_offsets
gl.store(c_ptrs, c, mask=c_mask)
```

**After:**
```python
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)
```

### A.3 Transformation checklist

For each `gl.load`:
- [ ] Separate `ptr + offsets` into scalar `base` and vector `offsets`
- [ ] Replace `gl.load(ptr, mask=m, other=v)` → `gl.amd.cdna3.buffer_load(ptr=base, offsets=offsets, mask=m, other=v)`
- [ ] Update loop increment: `base += stride` (not `ptr += stride`)

For each `gl.store`:
- [ ] Replace `gl.store(ptr, val, mask=m)` → `gl.amd.cdna3.buffer_store(ptr=base, offsets=offsets, stored_value=val, mask=m)`

### A.4 Verify correctness

```python
import torch, sys
sys.path.insert(0, '/home/leling/gfx9-gluon-tutorials/kernels/gemm/a16w16/v1_buffer_load')
from matmul_kernel import matmul as matmul_ref
# ... run your modified kernel and compare outputs ...

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
c_ref = matmul_ref(a, b)
c_new = your_matmul(a, b)
assert torch.allclose(c_ref, c_new, atol=1e-2, rtol=1e-2), "FAILED"
print("OK, max diff:", (c_ref - c_new).abs().max().item())
```

### A.5 Verify ASM improvement (optional)

```bash
TRITON_DUMP_BACKEND_IR=1 python3 <kernel.py> 2>&1 | grep -c "s_cbranch"        # should be ~0
TRITON_DUMP_BACKEND_IR=1 python3 <kernel.py> 2>&1 | grep -c "buffer_load_dwordx4"  # should be > 0
```

### A.6 Measure performance (Mode 1 — perf table)

Use `/kernel-perf-analysis` in **Mode 1** to capture TFLOPS, VGPRs, and average
kernel time before and after Step A. Mode 1 is triggered by mentioning "TFLOPS",
"perf table", or "benchmark":

```
/kernel-perf-analysis
Kernel file: <absolute path to kernel.py>
Mode hint: perf table
Label: step_a_buffer_load
```

The skill spawns two att-runner agents (kernel-trace + ATT) in parallel, then
prints a table:

```
| Version              | TFLOPS | VGPRs | Spills | MFMA Eff. | avg time  |
|----------------------|--------|-------|--------|-----------|-----------|
| step_a_buffer_load   |    118 |   200 |      0 |    57.98% | 795.88 us |
```

Expected: branch count drops from ~140 to ~4; latency improvement typically 5–15%
on long-K kernels.

**Fallback**: If no improvement, check whether `M * K * sizeof(fp16) > 4 GB`
(buffer ops require 32-bit offsets). For oversized tensors, remain on `gl.load`.

---

## Step B: Async Copy Through LDS (CDNA4 / gfx950 Only)

**Stop here if on gfx942.** Proceed to `/gemm-v3-lds-layout` instead.

### Why

The CDNA4 async copy DMA engine can transfer data from global memory directly into
LDS without occupying CU execution resources. This:
- Frees the CU to do other work while the DMA runs
- Is the prerequisite for overlapping memory and compute in `/gemm-v4-global-prefetch`
- Replaces the register-buffered load path entirely

In this version (v2), `wait_group(0)` keeps it synchronous — the overlap comes in v4.

### B.1 Add shared memory allocation (before the loop)

```python
# Start with trivial swizzle — tuned in /gemm-v3-lds-layout
sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)
```

### B.2 Replace buffer_load with async_copy in the loop

**Before (Step A / v1):**
```python
for k in range(0, gl.cdiv(K, BLOCK_K)):
    ga = gl.amd.cdna3.buffer_load(ptr=a_base, offsets=a_offsets, mask=mask_a, other=0.0)
    gb = gl.amd.cdna3.buffer_load(ptr=b_base, offsets=b_offsets, mask=mask_b, other=0.0)
    a = gl.convert_layout(ga, layout=dotOpLayoutA)
    b = gl.convert_layout(gb, layout=dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

**After (Step B / v2):**
```python
for k in range(0, gl.cdiv(K, BLOCK_K)):
    # DMA: global → LDS (non-blocking issue)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA, a_base, a_offsets, mask=offs_ak[None, :] < K - k * BLOCK_K, other=0.0
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB, b_base, b_offsets, mask=offs_bk[:, None] < K - k * BLOCK_K, other=0.0
    )
    gl.amd.cdna4.async_copy.commit_group()   # mark this DMA batch
    gl.amd.cdna4.async_copy.wait_group(0)    # wait for ALL batches (synchronous for now)

    # Read from LDS — layout conversion implicit
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA, dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB, dotOpLayoutB)

    acc = gl.amd.cdna3.mfma(a, b, acc)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

Note: `gl.convert_layout(ga, ...)` is removed — `load_shared_relaxed` reads directly
into the target layout.

### B.3 Verify correctness

```python
import torch, sys
sys.path.insert(0, '/home/leling/gfx9-gluon-tutorials/kernels/gemm/a16w16/v2_async_copy')
from matmul_kernel import matmul as matmul_ref
# Compare your kernel output against the reference

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
c_ref = matmul_ref(a, b)
c_new = your_matmul(a, b)
assert torch.allclose(c_ref, c_new, atol=1e-2, rtol=1e-2), "FAILED"
print("OK, max diff:", (c_ref - c_new).abs().max().item())
```

### B.4 Measure performance (Mode 1 — perf table)

Use `/kernel-perf-analysis` in **Mode 1** to confirm the kernel still runs
correctly and to establish a TFLOPS baseline for the async_copy version:

```
/kernel-perf-analysis
Kernel file: <absolute path to kernel.py>
Mode hint: perf table
Label: step_b_async_copy
```

**Expected behavior**: Step B alone may show equal or slightly worse performance
versus Step A because `wait_group(0)` is still synchronous. This is normal — the
structural value of Step B is unlocked in `/gemm-v4-global-prefetch` (double buffering
+ `wait_group(1)`).

### B.5 Check LDS bank conflicts (Mode 2 — counter collection)

The trivial `SwizzledSharedLayout(1, 1, 1, ...)` will likely cause LDS bank conflicts.
Use `/kernel-perf-analysis` in **Mode 2** to measure them before proceeding to
`/gluon-lds-opt`. Mode 2 is triggered by mentioning "bank conflict" or a counter name:

```
/kernel-perf-analysis
Kernel file: <absolute path to kernel.py>
Mode hint: bank conflict counter SQ_LDS_BANK_CONFLICT
Label: step_b_lds_conflicts
```

Expected output:
```
| Version              | SQ_LDS_BANK_CONFLICT | SQ_LDS_DATA_FIFO_FULL | Dispatches |
|----------------------|---------------------|-----------------------|------------|
| step_b_lds_conflicts |          12,345,678 |                     0 |         20 |
```

A non-zero `SQ_LDS_BANK_CONFLICT` confirms that `/gluon-lds-opt` should be applied next.

---

## Step C: Attention & Multi-Tile Patterns (CDNA4 / gfx950 Only)

Step B covers the canonical 2-tile GEMM (one A-tile, one B-tile, one commit
per iteration). Attention kernels (FlashAttention, MLA, sparse MLA, paged
attention) introduce three additional patterns that surface CDNA4 constraints
and async-pipeline subtleties not present in plain GEMM:

1. **More than two tiles per iteration** (e.g. K_main + K_rope + V or topk indices)
2. **Load-once-reuse-many operands** (e.g. Q, biases, RoPE tables) that don't move per iteration
3. **One source tile consumed by two MFMAs in different orientations** (e.g. K and V where V = trans(K))

The rules below apply to any attention kernel on gfx950. They are written as
generic patterns — substitute your own tile names.

### C.1 — The CDNA4 coalesced-write constraint (must-know rule)

When you switch a `buffer_load` to `async_copy.buffer_load_to_shared` and the
kernel fails to compile with:

```
error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
RuntimeError: failed to translate module to LLVM IR
```

…the most common cause is that the **source `BlockedLayout` does not match the
direct-to-LDS DMA's coalesced-write requirement**, so
`canCoalesceWriteIntoSharedMemory` rejects the lowering and the AMD
`amdg.buffer_load_to_local` op survives unlowered.

**The constraint** (CDNA4 only):

> The product of `size_per_thread[contig_dim]` and `threads_per_warp[contig_dim]`
> must equal the inner (contiguous) dim of the tile being DMA'd.

Equivalently: a single warp's coalesced span along the contiguous dim must
exactly cover one row (or column, depending on `order`) of the tile. Replication
along the contiguous dim is forbidden under-spans (warp covers less than one
row) are forbidden too.

Plus two hard widths from `supportsDirectToLdsLoadBitWidth`:

> Each thread's contiguous load must be **128 bits** or **32 bits** (no other widths).

For bf16/fp16 this means `size_per_thread[contig_dim] = 8` (128 bits) or `2`
(32 bits). For fp8/int8 → 16 or 4. Pick the wider option for max throughput.

**The adaptive layout pattern** (when the inner dim is a constexpr that varies
between kernel specializations):

```python
# Generic recipe — INNER_DIM is the contiguous dim of the tile being DMA'd.
# Keep size_per_thread[contig_dim] = 8 for bf16 (128-bit LDS write).
_tpw_contig: gl.constexpr = min(64, INNER_DIM // 8)
_tpw_other:  gl.constexpr = 64 // _tpw_contig         # warp = 64 threads on CDNA
blk_X: gl.constexpr = gl.BlockedLayout(
    size_per_thread = [..., 8],                       # 8 in the contig slot
    threads_per_warp = [..., _tpw_contig],            # adapt to INNER_DIM
    warps_per_cta = [..., 1],                         # or split to other dim
    order = [..., contig_dim_index_first],
)
```

Order of slots depends on `order`; place `8` and `_tpw_contig` in the slot
indexed by `order[0]` (the most-minor / contiguous dim).

**Validate it works**:
- Compile the kernel. If `unrealized_conversion_cast` reappears, dump IR with
  `MLIR_ENABLE_DUMP=1 python ... 2>/tmp/ir.txt` and check the failing op's
  tensor shape — the shape's inner dim almost always points at the offending
  layout.
- For shapes where `INNER_DIM // 8 > 64` you must increase `warps_per_cta` or
  drop to `size_per_thread[contig_dim] = 2` (32-bit LDS write); the constraint
  scales with warp size, not with tile size.

### C.2 — Multi-tile commit groups: pack tiles that advance together

Attention iterations often issue 2–3 async DMAs per step (e.g. K + V, or K_lora
+ K_rope + topk-prefetch). The async-copy machinery uses **groups**, not
individual operations:

- `commit_group()` packages every async copy issued since the last commit
  into one numbered group.
- `wait_group(N)` blocks until **at most N** groups remain in flight (older
  groups complete first; ordering is FIFO).

**Rule**: pack tiles that **advance synchronously each iteration into the
same group** — one `commit_group()` after issuing all of them. This way one
`wait_group(1)` retires the entire previous iteration's payload at once.

```python
# Pattern: issue N tiles, one commit, one wait per iteration
gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_a.index(buf), ...)
gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_b.index(buf), ...)
gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_c.index(buf), ...)
gl.amd.cdna4.async_copy.commit_group()           # one group covers a+b+c
# ... in next iteration, after issuing the new group:
gl.amd.cdna4.async_copy.wait_group(1)            # retires last iter's a+b+c
```

**Anti-pattern**: one commit per tile creates one group per tile and forces
finer-grain waits, which serializes the DMAs and burns extra `s_waitcnt`
cycles. Don't do it unless tiles really do advance independently (rare).

### C.3 — Multi-group prologues for load-once operands (e.g. Q in attention)

"Load-once" operands like Q (queries), biases, or RoPE rotation tables are
loaded in the prologue and reused for every iteration. **They should still
use async DMA** — overlap them with the first iteration's K-tile prefetch.

The prologue then has **two groups in flight**:

```
Group A (older): Q     — committed first
Group B (newer): K[0]  — committed second
```

After the K[0] commit, use `wait_group(1)` to retire **A** (Q) while leaving
**B** (K[0]) in flight. The first loop iteration issues K[1] as group C,
then `wait_group(1)` retires B. The pipeline is full from the start with
no stall on Q.

```python
# Prologue
gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_q, ...)  # Q
gl.amd.cdna4.async_copy.commit_group()                           # group A
gl.amd.cdna4.async_copy.buffer_load_to_shared(dest=smem_k.index(0), ...)  # K[0]
gl.amd.cdna4.async_copy.commit_group()                           # group B
gl.amd.cdna4.async_copy.wait_group(1)                            # A done; B in flight
Q_dot = smem_q.load(dot_op_layout)                               # convert once, keep in regs

# Main loop — wait_group(1) inside the loop now refers to "K[t] retired".
```

**Sizing**: keep load-once operands **single-buffered** (no need for `[2,...]`).
Small per-iter tiles can be double-buffered; the LDS budget rule (§ C.5) decides.

### C.4 — Reuse one shared buffer for two MFMA roles via `permute`

If a tile has shape `[M, N]` and you need to consume it as **opIdx=1 of a
[?, M] dot** (untransposed) and also as **opIdx=1 of a [N, ?] dot** in its
transpose, you do **not** need two shared buffers and you do **not** need a
second async DMA. The `memdesc.permute([1, 0])` operation is a **view** —
it reorders the descriptor's logical axes without moving data. The two
loads then read the same underlying LDS bytes through different addressing
patterns:

```python
buf = smem_X.index(cur_buf)
X_op_b   = buf.load(dot_b_layout_for_first_dot)               # [M, N] view
Xt_op_b  = buf.permute([1, 0]).load(dot_b_layout_for_second_dot)  # [N, M] view
```

This pattern is essential for FlashAttention-style kernels where K is used
both as `K^T` (for `S = Q @ K^T`) and as `V = trans(K^T)` (for `acc = P @ V`)
when KV are shared (MQA/MLA). It saves one DMA per tile per iteration —
the largest single win at the inner loop.

**Caveat**: the shared layout's swizzle parameters must work for **both**
read patterns. The compiler picks for the first consumer; verify with a
correctness test. If bank conflicts skyrocket on the second read, fall back
to two buffers.

### C.5 — LDS budget & double-buffering policy

Doubling a tile's shared-memory allocation (for prefetch) costs `2 ×
sizeof(tile)` LDS. The total budget is:

| GPU | LDS per CU |
|-----|------------|
| MI300X / MI308X (gfx942) | 64 KB |
| MI350X / MI355X (gfx950) | 160 KB |

**Policy** for an attention kernel:

1. **Double-buffer per-iteration tiles** (K, V, K_rope, K_lora, …) — they're
   typically small relative to the LDS budget, and the prefetch/compute
   overlap is the whole point of Step C.
2. **Single-buffer load-once tiles** (Q, biases) — they don't move per
   iteration, so a second buffer is wasted.
3. **Sum the LDS bytes** before adding double-buffering. If the kernel was
   close to the LDS limit before, `2 ×` may push it over and force you to
   shrink `BLOCK_K` / `TILE_K` or drop a tile to `vec=2` (32-bit LDS path).

For a sparse-MLA-style kernel at typical production shapes, ~100 KB LDS
usage is normal on gfx950 and well within budget.

### C.6 — Pre-flight checklist before the first compile

- [ ] Each tile DMA'd to LDS satisfies the C.1 coalesced-write constraint.
- [ ] Each tile's `size_per_thread[contig_dim]` is `8` (128-bit) or `2`
  (32-bit) for bf16/fp16; `16` or `4` for fp8/int8.
- [ ] Tiles that advance together are packed into one `commit_group()`.
- [ ] Per-iteration tiles use `[2, ...]` shared allocations (`smem.index(buf)`).
- [ ] Load-once tiles use `[...]` shared allocations (no leading `2`).
- [ ] Total LDS ≤ 160 KB (gfx950) / 64 KB (gfx942).
- [ ] If reusing one buffer for transposed dot operand, `permute([1,0]).load`
  is in place and the swizzle layout is compatible with both reads.

### C.7 — Diagnosis playbook for async-copy failures

| Symptom | Probable cause | Fix |
|---------|----------------|-----|
| `LLVM Translation failed for operation: builtin.unrealized_conversion_cast` near a tile's shape | C.1 coalesced-write constraint violated for that tile | Adapt `threads_per_warp` per § C.1 (`min(warp_size, INNER_DIM // 8)` recipe) |
| `unrealized_conversion_cast` on a scalar/broadcast operand | Mask/offset broadcast pattern not supported by DMA | Materialize the mask explicitly with the same blocked layout as the offsets |
| Compiles, runs, wrong numerics | `wait_group(N)` not waiting for the right group | Walk the group accounting: count commits in prologue + loop, ensure compute reads happen *after* the corresponding wait |
| Compiles, runs, slower than sync version | All async copies committed individually (one per call) | Pack synchronously-advancing tiles into one `commit_group` per iteration (§ C.2) |
| Occupancy drops sharply after Step C | LDS budget exceeded, fewer waves per CU | Recheck § C.5; drop double-buffering on the smallest-benefit tile, or shrink TILE_K |
| MFMA efficiency below expectations after Step C | Bank conflicts on the doubled-buffered shared layout (often surfaces only after async pattern lands) | Run `/lds-bank-conflict`; if non-zero, apply `/gluon-lds-opt` |

**Best diagnostic commands**:
```bash
# Dump every IR after every pass — lets you find which pass left the cast.
MLIR_ENABLE_DUMP=1 python <kernel.py> 2>/tmp/ir.txt
grep -B2 -A2 "unrealized_conversion_cast" /tmp/ir.txt | head

# Look for the failing tensor shape (often points straight at the bad layout).
grep "unrealized_conversion_cast.*tensor<" /tmp/ir.txt | head
```

### C.8 — Verify and measure (Mode 1 + Mode 2)

After applying Step C patterns:

```
/kernel-perf-analysis
Kernel file: <absolute path>
Mode hint: perf table + bank conflict counter SQ_LDS_BANK_CONFLICT
Label: step_c_attention_async
```

Expected wins from Step C on gfx950 attention kernels (typical):

- **Wait-counter stall reduction**: 20–35% per loop iteration (the dominant cost).
- **MFMA efficiency lift**: +10–15 percentage points.
- **End-to-end speedup vs. sync `buffer_load` baseline**: 1.10×–1.20×.

If you see less, walk § C.7 — most "no-op" results come from over-committing
groups (one per tile) rather than packing them.

---

## Summary of Changes

| | Step A (buffer_load) | Step B (async_copy, GEMM) | Step C (multi-tile / attention) |
|-|---------------------|---------------------------|---------------------------------|
| Load API | `gl.amd.cdna3.buffer_load` | `gl.amd.cdna4.async_copy.buffer_load_to_shared` | Same as B + multi-tile |
| Data destination | Registers | LDS | LDS (often double-buffered) |
| Synchronization | Implicit | `commit_group` + `wait_group(0)` | `commit_group` per group; `wait_group(N)` for pipeline |
| LDS read | — | `load_shared_relaxed` | `smem.index(buf).load(dot_layout)` and `permute` views |
| `convert_layout` needed | Yes (explicit) | No (implicit in `load_shared_relaxed`) | No |
| Tiles per iteration | 1 (per A or B) | 1 (per A or B) | N (e.g. K + V + index, packed in one group) |
| GPU requirement | CDNA3 + CDNA4 | CDNA4 only | CDNA4 only |
| Primary benefit | Eliminate branches | Enables async pipeline | Hides multi-tile DMA latency in attention loops |
| Critical rule | None | `wait_group(0)` is still synchronous | C.1 coalesced-write constraint; group packing (C.2) |

