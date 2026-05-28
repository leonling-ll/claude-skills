---
name: gluon-lds-opt
description: >
  Fix LDS (Local Data Share) bank conflicts in a Gluon kernel (GEMM or attention)
  that loads tiles into shared memory via async_copy or buffer_load. Symptoms:
  high SQ_LDS_BANK_CONFLICT hardware counter, high-cycle s_waitcnt lgkmcnt(0)
  before MFMA in ATT traces, or ds_read instructions on the critical path in the
  amdgcn ISA. Three strategies: (1) swizzling — change SwizzledSharedLayout
  parameters from trivial (1,1,1) to bank-conflict-free (8,1,8); (2) full
  padding — use PaddedSharedLayout with DistributedLinearLayout for global loads;
  (3) compiler-matching padding via `PaddedSharedLayout.with_identity_for(...)`
  for gfx950 + async_copy + bf16/fp16/fp8/i8 + MFMA, mirroring what the AMD
  Triton compiler picks. Bank conflicts can reduce LDS throughput by 8–32x.
  Applies to both CDNA3 (gfx942) and CDNA4 (gfx950); padded path (3) is gfx950
  only. Use /lds-bank-conflict to measure conflicts before and after. Trigger
  for any mention of LDS bank conflicts, ds_read stalls, lgkmcnt stalls, or
  SwizzledSharedLayout in a Gluon kernel.
  Usage: /gluon-lds-opt
---

# Gluon LDS Bank-Conflict Optimization

Fix LDS bank conflicts in a Gluon GEMM kernel that loads tiles via `async_copy` or
`buffer_load`. The root cause is a trivial `SwizzledSharedLayout(1,1,1,...)` that maps
threads to overlapping LDS banks, serializing reads and writes.

## Background: LDS Bank Conflicts on CDNA3/4

AMD CDNA3 GPUs have **32 LDS banks** and CDNA4 GPUs have **64 LDS banks**, each 4 bytes wide. 
When multiple threads in the same wavefront access the **same bank** (but different addresses), 
the accesses are serialized — a 32-way bank conflict takes 32x longer than a conflict-free access.

With `SwizzledSharedLayout(1, 1, 1, ...)` the stored elements are nearly sequential
in LDS, causing wavefront-level accesses to land on the same banks repeatedly.

**Why it hurts so much:** With 64 threads per wavefront and 32 banks, a naive layout
causes ~2x conflicts on average; worst-case (all threads hitting one bank) is 64x.
The right swizzle or padding pattern ensures each of the 64 threads hits a different
bank, giving up to 32x LDS throughput improvement over the trivially conflicted case.

The fix: rearrange the LDS storage so each thread in a wavefront hits a different bank.

## Step 0: Check GPU Platform

```bash
python3 -c "import torch; props = torch.cuda.get_device_properties(0); print(props.gcnArchName)"
```

This optimization applies to **both gfx942 (CDNA3) and gfx950 (CDNA4)**. The exact
layout parameters below are validated for both platforms.

## Step 1: Diagnose LDS Bank Conflicts

There are 3 options to measure LDS bank conflicts. Take Option A as default, if fail, them move to Option B and then C.
After finishing, print a brief bank conflict status before take following steps.

### Option A: Use the /lds-bank-conflict skill

```bash
# Invoke the dedicated skill to measure SQ_LDS_BANK_CONFLICT counts
/lds-bank-conflict python3 <your_kernel.py>
```

### Option B: Manual rocprofv3 counter collection

```bash
touch /tmp/t0
rocprofv3 --counter-collection "SQ_LDS_BANK_CONFLICT,SQ_LDS_DATA_FIFO_FULL" \
  -- python3 <your_kernel.py> 2>&1
```

Expected output for a conflicted kernel: `SQ_LDS_BANK_CONFLICT > 0` (often millions).
After fix: close to 0.

### Option C: Check ISA for ds_read stall patterns

```bash
# Find the compiled amdgcn ISA for your kernel
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5

# Look for lgkmcnt(0) before mfma — means LDS read is on the critical path
grep -c "s_waitcnt lgkmcnt(0)" <path>.amdgcn
```

A high count of `s_waitcnt lgkmcnt(0)` immediately before `v_mfma_*` instructions
indicates the compiler cannot pipeline LDS reads — the MFMA is stalling waiting for
`ds_read` to complete. This is a strong signal of bank-conflict-induced serialization.

Also check in ATT traces (via `/kernel-trace-analysis`) for high-cycle `ds_read`
instructions with `lgkmcnt` stalls shown in the timeline.

## Step 2: Choose a Strategy

Three approaches are available; pick by GPU + workload:

| Strategy | API | Ease | When to use |
|----------|-----|------|-------------|
| **Swizzling** (Step 3a) | `SwizzledSharedLayout(8, 2, 8, ...)` | Easy (change 3 numbers) | gfx942 (always); gfx950 GEMM with sync `buffer_load`; any tile where padded path is rejected (see § 3c) |
| **Full padding** (Step 3b) | `PaddedSharedLayout` + `DistributedLinearLayout` | Most verbose | When you want explicit control over both the global load layout and the LDS layout |
| **Compiler-matching padding** (Step 3c) | `PaddedSharedLayout.with_identity_for([[interval, padding]], shape, order)` | Easy (one constructor call) | **gfx950 only**: async_copy + bf16/fp16/fp8/i8 + MFMA dot operand + kWidth ∈ {4,8,16} + mfmaNonKDim ∈ {16,32}. Mirrors the AMD compiler's choice. Hand-written attention/MLA kernels in Gluon almost always want this. |

**Decision tree:**
- gfx942 (CDNA3, MI300X) → Swizzling (Step 3a). The compiler never picks padded on this arch (the LDS budget is too tight at 64 KB).
- gfx950 (CDNA4, MI350) + sync load (no async_copy) → Swizzling.
- gfx950 + async_copy + bf16/fp16/fp8/i8 + tile inner dim ≥ `paddingInterval` (see § 3c) → Compiler-matching padding (Step 3c). Single best win in attention kernels.
- gfx950 + async_copy + tile inner dim < `paddingInterval` (e.g., RoPE-like 64-element tiles) → Swizzling (the padded path is rejected by the compiler's bank-conflict heuristic anyway).
- Need explicit `DistributedLinearLayout` for the global load (rare outside hand-tuned GEMM) → Full padding (Step 3b).

## Step 3a: Apply Swizzling (Simpler)

Change the `SwizzledSharedLayout` parameters from trivial to proper values derived
from the hardware bank structure, tile shape, and element type.

### SwizzledSharedLayout Parameter Derivation Rules

`SwizzledSharedLayout(vec, perPhase, maxPhase, order=[...])` applies an XOR-based
address permutation: element at `(row, col)` is stored at address
`((col / vec) ^ (row / perPhase) % maxPhase) * vec + (col % vec)`.

**Rule 1 — Determine `order` from operand role:**
- Operand A (K-major read): fastest dim is K → `order=[1, 0]` (K is dim 1)
- Operand B (K-major read): fastest dim is K → `order=[0, 1]` (K is dim 0)
- The swizzle only helps when the fastest dim (order[0]) is the K dimension.
  If the layout is already non-K-contiguous, no swizzling is needed (`vec=perPhase=maxPhase=1`).

**Rule 2 — Compute `vec` (vector size for ds_read):**
- `vec = min(kWidth * elemBitWidth, 128) / elemBitWidth`
  - `kWidth` = number of K elements per thread per instruction (from the MFMA tile)
  - cap at 128 bits because `ds_load` max granularity is 128 bits
- For fp16 (`elemBitWidth=16`): `vec = min(kWidth * 16, 128) / 16`
  - kWidth=8 → vec=8; kWidth=4 → vec=4
- For fp8 (`elemBitWidth=8`): `vec = min(kWidth * 8, 128) / 8`
  - kWidth=16 → vec=16

**Rule 3 — Compute `perPhase` (rows sharing the same XOR pattern):**
- `elemsPerBankRow = (numBanks * 32) / elemBitWidth`
  - CDNA3/gfx942: `numBanks=32` → elemsPerBankRow = `1024 / elemBitWidth`
  - CDNA4/gfx950: `numBanks=64` → elemsPerBankRow = `2048 / elemBitWidth`
- `innerDimLength = shape[order[0]]` (length of fastest-changing tile dimension)
- `perPhase = max(1, elemsPerBankRow / innerDimLength)`
- Example: fp16, CDNA3, K-dim=64 → elemsPerBankRow=64, perPhase=1
- Example: fp16, CDNA3, K-dim=128 → elemsPerBankRow=64, perPhase=1 (already >1)
- Example: fp16, CDNA4, K-dim=64 → elemsPerBankRow=128, perPhase=2

**Rule 4 — Compute `maxPhase` (period of the XOR pattern):**
- `simdWidth = 16` (MFMA instruction M/N tile size; 4 for 4x4 MFMA)
- `maxPhase = max(1, min(simdWidth / perPhase, innerDimLength / vec))`
- For MFMA 4x4 variant: cap `maxPhase=4`
- Example: fp16, CDNA3, K=64, vec=8, perPhase=1, simdWidth=16 → maxPhase=min(16,8)=8
- Example: fp16, CDNA4, K=64, vec=8, perPhase=2, simdWidth=16 → maxPhase=min(8,8)=8

**Rule 5 — Special cases that skip swizzling:**
- Scale operands (operandIdx ≥ 2): always use `(1, 1, 1)` — no swizzle
- Non-K-contiguous layouts on CDNA3: use `(1, 1, 1)` — different banks naturally
- Non-K-contiguous layouts on CDNA4 with 8-bit or 16-bit elements: still swizzle

### Worked Example: Standard fp16 GEMM (BLOCK_M=256, BLOCK_K=64, BLOCK_N=256)

MFMA 16x16 → kWidth=4 per thread (for fp16). CDNA3 (gfx942, 32 banks):

```
For A tile (256×64, order=[1,0], K along dim 1):
  vec       = min(4 * 16, 128) / 16 = min(64, 128) / 16 = 4?
              ... typical MFMA reads 8 fp16 → vec=8
  elemsPerBankRow = (32 * 32) / 16 = 64
  innerDimLength  = 64   (K dim)
  perPhase  = max(1, 64 / 64) = 1
  maxPhase  = max(1, min(16 / 1, 64 / 8)) = max(1, min(16, 8)) = 8

  → SwizzledSharedLayout(8, 1, 8, order=[1, 0])   ← many kernels use (8,2,8)
    because perPhase=2 groups rows to match 128-bit-wide bank rows on some configs

For B tile (64×256, order=[0,1], K along dim 0):
  Same arithmetic but inner dim is now dim 0 (length 64)
  → SwizzledSharedLayout(8, 1, 8, order=[0, 1])
```

> **Practical note:** `(8, 2, 8)` is the canonical bank-conflict-free setting that
> Triton's compiler emits for fp16 MFMA 16x16 on CDNA3 with K=64. Use it as the
> starting point; tune `perPhase` (1 or 2) if profiling shows residual conflicts.

### Code Changes

**Before (trivial, conflict-prone):**
```python
sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
```

**After (bank-conflict-free, fp16 MFMA 16x16 on CDNA3 with K=64):**
```python
sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
```

**Quick-reference table for common fp16 configurations:**

| GPU    | K-dim | vec | perPhase | maxPhase |
|--------|-------|-----|----------|----------|
| CDNA3  | 32    |   8 |        2 |        4 |
| CDNA3  | 64    |   8 |        1 |        8 |
| CDNA3  | 128   |   8 |        1 |        8 |
| CDNA4  | 32    |   8 |        4 |        4 |
| CDNA4  | 64    |   8 |        2 |        8 |
| CDNA4  | 128   |   8 |        1 |        8 |

Also add a compiler hint to help eliminate dead code in the loop:
```python
max_iter = gl.cdiv(K, BLOCK_K)
gl.assume(max_iter > 0)   # ADD THIS before the loop

for k in range(0, max_iter):   # use max_iter instead of gl.cdiv(K, BLOCK_K)
    ...
```

No other code changes are needed for swizzling — the layout controls how data is
physically stored in LDS, transparently to the rest of the kernel.

## Step 3b: Apply Padding (Alternative)

Padding requires changing both the global load layout and the shared memory layout.
The parameters below are for `BLOCK_M=256, BLOCK_K=64, BLOCK_N=256`.

### Change global load layouts to DistributedLinearLayout

**For A tile (BLOCK_M x BLOCK_K = 256 x 64):**
```python
gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]],
    lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
    warp_bases=[[1, 0], [2, 0]],
    block_bases=[],
    shape=[BLOCK_M, BLOCK_K],
)
```

**For B tile (BLOCK_K x BLOCK_N = 64 x 256):**
```python
gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 128]],
    lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
    warp_bases=[[0, 1], [0, 2]],
    block_bases=[],
    shape=[BLOCK_K, BLOCK_N],
)
```

### Change shared layouts to PaddedSharedLayout

**For A tile:**
```python
sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
    [[512, 16]],
    [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32],
        [16, 0], [32, 0], [64, 0],
        [1, 0], [2, 0], [4, 0], [8, 0], [128, 0],
    ],
    [],
    [BLOCK_M, BLOCK_K],
)
```

**For B tile:**
```python
sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
    [[512, 16]],
    [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 16], [0, 32], [0, 64],
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 128],
    ],
    [],
    [BLOCK_K, BLOCK_N],
)
```

### Update arange calls to use the new layout slices

```python
offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
```

## Step 3c: Compiler-matching Padding (gfx950 Only, Recommended for Attention)

When the AMD Triton compiler picks `PaddedSharedEncoding` for an analogous
Triton kernel, you should pick the same in Gluon. This step gives the rules
for when the compiler picks padded over swizzled, the formula for the
parameters, and the simple Gluon API to construct it.

### When the compiler picks padded over swizzled

`composePaddedLayout` in the AMD backend tries the padded path first; falls
back to swizzled if padded returns null. Padded is selected **only** when
**all** of the following hold:

1. `arch == gfx950` (CDNA4) — gfx942 always returns null.
2. `useAsyncCopy == true` (the load is an `async_copy.buffer_load_to_shared`,
   not a sync `buffer_load`).
3. Parent encoding is `AMDMfmaEncoding` (the LDS load feeds an MFMA dot operand).
4. Tensor rank is 2.
5. `elemByteWidth ∈ {1, 2}` (fp8 / int8 / fp16 / bf16 — not fp32/i32).
6. `mfmaNonKDim ∈ {16, 32}`.
7. `kWidth ∈ {4, 8, 16}` (dot-operand `k_width`).
8. Operand index < 2 (no scaled-dot scales).
9. The bank-conflict heuristic accepts the resulting layout (predicts
   ≤ 0 conflicts for `useDsReadB128`, ≤ 2-way for `useDsReadB64Tr`).

If **any** of those fail (most commonly: small inner dim → heuristic rejects;
sync load → not async; fp32 → not in elem-byte-width set), the compiler
falls back to swizzled and so should you.

### Padding interval & amount formula

For the case that triggers most often in attention (K-contig, kWidthBytes==16,
i.e. bf16/fp16 with kWidth=8):

```text
paddingInterval = warpSize * (16 / elemBytes)
                = 64 * (16 / elemBytes)
                = 512 elements for bf16/fp16
                = 1024 for fp8/int8

padding (K-contig, useDsReadB128, kWidthBytes==16):
                = (mfmaNonKDim == 16) ? 2*kWidth : kWidth
                = 16 for nkdim=16 + kWidth=8
                =  8 for nkdim=32 + kWidth=8

padding (!K-contig, useDsReadB64Tr, kWidthBytes >= 8):
                = (mfmaNonKDim == 16) ? 16 : 32

padding (other K-contig kWidths):
                = 8 / elemBytes
                = 4 for bf16/fp16
                = 8 for fp8/int8
```

> Reference: `triton_amd_shared_encoding_rules.md` § 4.2.1 (constructed by
> `composePaddedLayoutForAsyncCopyCDNA4` in `Utility.cpp:149-353`). The doc
> also covers the gfx1250/WMMA variant (rarely hit in CDNA4 work).

### The "innerDim < paddingInterval" rejection rule

The bank-conflict heuristic in the compiler refuses to emit a padded layout
when the tile's contiguous inner dim is smaller than `paddingInterval`. In
practice that means:

- bf16/fp16 + tile inner dim < 512 elements → padded rejected → use swizzled
- fp8/int8 + tile inner dim < 1024 elements → padded rejected → use swizzled

Common attention case: **RoPE-style sub-tiles with inner dim 64** (Q_rope,
K_rope) always go swizzled even on gfx950, while the **main D_v / D_k tiles
with inner dim ≥ 512** go padded. Mirror this — don't try to force padding
on the small tiles.

### Gluon API

Use the simple `with_identity_for` constructor — it builds the padded layout's
linear component as the identity remap (no row stagger). This matches what
the compiler emits for the canonical case:

```python
sh_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[interval, padding]],   # one or more interval/padding pairs
    [outerDim, innerDim],    # per-buffer 2D shape; for double-buffered
                             # [2, outerDim, innerDim] allocations, pass the
                             # 2D slice shape, not the 3D allocation shape
    order,                   # [contigDim, otherDim] — same as Swizzled
)
```

Concrete recipes for gfx950 + bf16/fp16 + MFMA + async_copy + kWidth=8:

```python
# K-contig, mfmaNonKDim=16 (most common): use [[512, 16]]
sh_main: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[512, 16]], [outerDim, innerDim], [contigDim, otherDim],
)

# K-contig, mfmaNonKDim=32: use [[512, 8]]
sh_main_nkdim32: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[512, 8]], [outerDim, innerDim], [contigDim, otherDim],
)

# Small inner dim (< 512 for bf16) — DO NOT pad, use swizzled instead
sh_small: gl.constexpr = gl.SwizzledSharedLayout(
    vec=8, per_phase=2, max_phase=8, order=[contigDim, otherDim],
)
```

### Reuse one padded buffer for two MFMA operand roles

In attention, K is often consumed twice per tile: once as `K^T` for the S-dot
and once as `V = trans(K^T)` for the acc-dot, via a `smem.permute([1,0])`
view. The padded layout works for both consumers as long as the kWidths are
compatible (typically S-dot uses `kWidth=8`, acc-dot uses `kWidth=4`, both
in the {4,8,16} set). Empirically this is conflict-free — no need to
allocate two separate buffers like the Triton compiler does. See the
`gluon-mm-inst-opt` skill § C.4 for the permute-view pattern.

### LDS budget impact

Padding adds bytes. Per-buffer overhead = `(elements / interval) × padding × elemBytes`.

| Tile shape (bf16) | Interval/padding | Bytes added/buf | Note |
|-------------------|------------------|-----------------|------|
| `[64, 512]` Q_lora | `[[512, 16]]` | 2 KB | 32 K elements / 512 = 64 pads of 16 elems × 2 B |
| `[2, 512, 16]` K_lora dbuf | `[[512, 16]]` | 2 × 1 KB = 2 KB | per-buffer ~1 KB extra |

Total padding cost for a typical Gluon attention kernel: ~5 KB. Negligible
on gfx950 (160 KB LDS). On gfx942 the padded path is disabled at the
compiler level for exactly this reason — don't try to back-port it.

### Worked example outcome (DSA forward kernel)

Switching the two D_v-inner tiles (Q_lora `[64, 512]`, K_lora `[512, 16]`)
from `SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16)` to
`PaddedSharedLayout.with_identity_for([[512, 16]], …)` while leaving the
two D_rope-inner tiles swizzled (innerDim=64 below the 512-element
threshold) gave **+10% end-to-end speedup** on the production attention
shape (TOPK=1024, MI350X), matching the gain from the compiler's own
choice. No `lds-bank-conflict` regression on the swizzled small tiles.

### Pitfalls

1. **Don't use `[[512, 16]]` for fp8/int8** — `paddingInterval` is 1024 for
   1-byte elements; use `[[1024, 32]]` per the formula.
2. **Don't pad sync loads** — `async_copy.buffer_load_to_shared` is the
   trigger; sync `buffer_load` into a padded layout has no compiler path
   to consume the padding correctly.
3. **Don't pad small tiles** — if you copy `[[512, 16]]` to a tile with
   inner dim 64, the layout will compile but you waste LDS without
   improving bank distribution. Use Swizzled for those.
4. **Don't bother on gfx942** — the compiler returns null for the padded
   path on CDNA3 because its 64 KB LDS budget is too tight. Stick with
   Swizzled there.
5. **Verify with the bank-conflict counter** — `with_identity_for` does
   not include the row-stagger that the compiler-emitted padded layout
   uses internally. For the canonical bf16 / kWidth=8 / nkdim=16 case
   this still gives 0 conflicts; for unusual configs you may need to
   construct `PaddedSharedLayout(...)` explicitly with custom
   `offset_bases` to match the compiler's row stagger.

## Step 4: Verify Correctness

Replace the stub paths below with your actual kernel module paths:

```python
import torch, sys

# --- Fill in your kernel paths ---
# sys.path.insert(0, '<path_to_your_baseline_kernel_dir>')
# from matmul_kernel import matmul as matmul_baseline
# sys.path.insert(0, '<path_to_your_optimized_kernel_dir>')
# from matmul_kernel import matmul as matmul_opt
# ---------------------------------

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
c_baseline = matmul_baseline(a, b)
c_opt = matmul_opt(a, b)
assert torch.allclose(c_baseline, c_opt, atol=1e-2, rtol=1e-2), "FAILED"
print("Correctness OK, max diff:", (c_baseline - c_opt).abs().max().item())
```

## Step 5: Measure Performance and Bank Conflict Reduction (Mode 2 — counter collection)

Use `/kernel-perf-analysis` in **Mode 2** to measure `SQ_LDS_BANK_CONFLICT` before
and after the layout change. Mode 2 is triggered by mentioning "bank conflict",
"counter", or a hardware counter name:

**Before fix:**
```
/kernel-perf-analysis
Kernel file: <absolute path to baseline_kernel.py>
Mode hint: bank conflict counter SQ_LDS_BANK_CONFLICT SQ_LDS_DATA_FIFO_FULL
Label: before_swizzle
```

**After fix:**
```
/kernel-perf-analysis
Kernel file: <absolute path to optimized_kernel.py>
Mode hint: bank conflict counter SQ_LDS_BANK_CONFLICT SQ_LDS_DATA_FIFO_FULL
Label: after_swizzle
```

Expected output showing the improvement:
```
| Version        | SQ_LDS_BANK_CONFLICT | SQ_LDS_DATA_FIFO_FULL | Dispatches |
|----------------|---------------------|-----------------------|------------|
| before_swizzle |          12,345,678 |                     0 |         20 |
| after_swizzle  |                   0 |                     0 |         20 |
```

Print the before/after table and summarize the conflict reduction.

Then also run **Mode 1** to confirm the TFLOPS improvement matches expectations:

```
/kernel-perf-analysis
Kernel file: <absolute path to optimized_kernel.py>
Mode hint: perf table benchmark
Label: after_swizzle
```

**Expected improvement:**
- Significant reduction in `SQ_LDS_BANK_CONFLICT` counter (often 10-100x reduction)
- Reduced `lgkmcnt` stall cycles visible in ATT trace
- Measurable kernel speedup (5-30% end-to-end depending on how LDS-bound the kernel was)

**Performance gain reasoning:** The trivial `SwizzledSharedLayout(1,1,1)` causes 32-way
bank conflicts because consecutive thread indices in a wavefront all map to the same
4-byte LDS bank after the address wraps modulo 32. With 64 threads per wavefront and
32 banks, on average every pair of threads conflicts, serializing LDS reads 2x. In the
worst case (all threads hitting one bank), throughput drops 64x. Proper swizzle or
padding parameters ensure each of the 64 threads in a wavefront hits a unique bank,
restoring full LDS throughput — up to 32x improvement over the worst-case conflicted
layout.

## Step 6: Fallback

If neither swizzling nor padding improves performance:
1. The kernel may already be compute-bound (MFMA limited), not LDS-limited; confirm
   by checking that `SQ_LDS_BANK_CONFLICT` is already near zero before your change
2. Restore the original layout and document the finding for the next optimization step

## Key Differences Between the Two Strategies

- **Swizzling**: only 2 lines change (layout params); transparent to the rest of the kernel
- **Padding**: requires `DistributedLinearLayout` for global load + `PaddedSharedLayout`
  for LDS; more lines to change but more explicit control
- Both eliminate bank conflicts for `BLOCK_M=256, BLOCK_K=64, BLOCK_N=256`
- For other tile sizes, the padding parameters must be re-derived; swizzling (8,2,8)
  generalizes better across common GEMM tile shapes

## Why Bank Conflicts Matter for GEMM

In a 256x64 tile with 4 warps:
- Each wavefront has 64 threads
- Each thread loads 8 fp16 elements = 16 bytes
- Trivial layout: consecutive rows map to the same 4-byte bank — 32-way conflicts on
  each `ds_read`
- Proper layout: each thread in a wavefront hits a unique bank — 1x throughput

