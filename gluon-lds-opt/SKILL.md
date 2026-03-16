---
name: gluon-lds-opt
description: >
  Fix LDS (Local Data Share) bank conflicts in a Gluon GEMM kernel that loads tiles
  into shared memory via async_copy or buffer_load. Symptoms: high SQ_LDS_BANK_CONFLICT
  hardware counter, high-cycle s_waitcnt lgkmcnt(0) before MFMA in ATT traces, or
  ds_read instructions on the critical path in the amdgcn ISA. Two strategies:
  (1) swizzling — change SwizzledSharedLayout parameters from trivial (1,1,1) to
  bank-conflict-free (8,2,8); (2) padding — use PaddedSharedLayout with
  DistributedLinearLayout for global loads. Bank conflicts can reduce LDS throughput
  by 8–32x and dominate kernel runtime. Applies to both CDNA3 (gfx942) and CDNA4
  (gfx950). Use /lds-bank-conflict to measure conflicts before and after. Trigger
  for /gemm-v3-lds-layout requests, or any mention of LDS bank conflicts, ds_read
  stalls, lgkmcnt stalls, or SwizzledSharedLayout in a Gluon kernel.
  Usage: /gluon-lds-opt
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Gluon LDS Bank-Conflict Optimization

Fix LDS bank conflicts in a Gluon GEMM kernel that loads tiles via `async_copy` or
`buffer_load`. The root cause is a trivial `SwizzledSharedLayout(1,1,1,...)` that maps
threads to overlapping LDS banks, serializing reads and writes.

## Background: LDS Bank Conflicts on CDNA3/4

AMD CDNA GPUs have **32 LDS banks**, each 4 bytes wide. When multiple threads in the
same wavefront access the **same bank** (but different addresses), the accesses are
serialized — a 32-way bank conflict takes 32x longer than a conflict-free access.

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

## Step 1: Diagnose Bank Conflicts

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

Two equivalent approaches are available:

| Strategy | API | Ease | Notes |
|----------|-----|------|-------|
| **Swizzling** | `SwizzledSharedLayout(8, 2, 8, ...)` | Easy (change 3 numbers) | Works for standard GEMM layouts |
| **Padding** | `PaddedSharedLayout` + `DistributedLinearLayout` | More verbose | More explicit, matches hardware exactly |

For most GEMM kernels with `BLOCK_M=256, BLOCK_K=64, BLOCK_N=256`, **swizzling is simpler**.
Padding is preferred when you need more control over the load layout or when swizzling
does not fully eliminate conflicts in your specific tile shape.

## Step 3a: Apply Swizzling (Simpler)

Change the `SwizzledSharedLayout` parameters from trivial to proper:

**Before (trivial, conflict-prone):**
```python
sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
```

**After (bank-conflict-free):**
```python
sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])
```

Also add a compiler hint to help eliminate dead code in the loop:
```python
max_iter = gl.cdiv(K, BLOCK_K)
gl.assume(max_iter > 0)   # ADD THIS before the loop

for k in range(0, max_iter):   # use max_iter instead of gl.cdiv(K, BLOCK_K)
    ...
```

No other code changes are needed for swizzling — the layout controls how data is
physically stored in LDS, transparently to the rest of the kernel.

**Why SwizzledSharedLayout(8, 2, 8) works:** The three parameters encode a permutation
that maps each thread in a wavefront to a unique 4-byte LDS bank. With trivial (1,1,1),
consecutive thread indices map to consecutive addresses which collide on the same bank
after wrapping at 32. With (8,2,8), the XOR-based swizzle spreads threads across all 32
banks, eliminating conflicts entirely for the standard tile shapes.

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

## Step 5: Measure Performance and Bank Conflict Reduction

```bash
# Before fix — measure conflicts on your original kernel
/lds-bank-conflict python3 <baseline_kernel.py>

# After fix — measure on the optimized kernel
/lds-bank-conflict python3 <optimized_kernel.py>

# Also compare wall-clock timing via rocprofv3 stats
rocprofv3 --stats -- python3 <optimized_kernel.py>
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
1. Check if the bottleneck is elsewhere — use `/kernel-trace-analysis` to identify the
   actual bottleneck instruction category
2. The kernel may already be compute-bound (MFMA limited), not LDS-limited; confirm
   by checking that `SQ_LDS_BANK_CONFLICT` is already near zero before your change
3. Restore the original layout and document the finding for the next optimization step

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

The `/lds-optimization` skill provides deeper analysis of LDS usage patterns if needed.
