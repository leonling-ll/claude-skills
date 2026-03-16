---
name: gluon-pipeline-opt
description: >
  Apply pipeline optimizations to a Gluon GEMM kernel that uses async_copy to LDS
  on CDNA4 (gfx950/MI350): two progressive stages to hide global and local memory
  latency. Apply Stage 1 (global prefetch / double buffering) when ATT trace or
  amdgcn shows s_waitcnt vmcnt(0) stalls before MFMA — indicating DMA latency is
  exposed. Apply Stage 2 (local prefetch) when lgkmcnt stalls remain after Stage 1
  — indicating ds_read latency on the critical path. Stage 1: allocate nBuffers=2
  shared memory, issue next iteration's DMA concurrently with current MFMA using
  wait_group(1). Stage 2: extend prologue to pre-load two tiles, issue MFMA first
  each iteration (register data already ready), then DMA and ds_read for the next
  iteration — achieving three-way overlap of DMA/ds_read/MFMA. Both stages require
  CDNA4; not applicable on gfx942 (CDNA3). Trigger for global prefetch, double
  buffering, local prefetch, LDS read overlap, three-way pipeline, vmcnt stalls,
  lgkmcnt stalls, or hiding memory latency in Gluon GEMM kernels.
  Usage: /gluon-pipeline-opt
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Gluon Pipeline Optimization: Global + Local Prefetch

Two progressive pipeline stages that overlap global DMA, LDS reads, and MFMA compute
to hide memory latency. Apply in order — global prefetch first, then local prefetch.

**Both stages require CDNA4 (gfx950/MI350).** Check the GPU before proceeding.

---

## Step 0: Check GPU Platform

```bash
python3 -c "import torch; props = torch.cuda.get_device_properties(0); print(props.gcnArchName)"
```

- `gfx950` → CDNA4 (MI350): both stages applicable, proceed.
- `gfx942` → CDNA3 (MI300X/MI308X): `async_copy` not available — **stop**. These
  optimizations do not apply on CDNA3. Document this and return to the caller.

---

## Step 0b: Identify Bottleneck from amdgcn

Locate the compiled kernel ISA to confirm which stall type dominates before
choosing which stage to apply:

```bash
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5
# Look for s_waitcnt vmcnt(0) before mfma → DMA latency exposed → apply Stage 1
grep -c "s_waitcnt vmcnt(0)" <path>.amdgcn
# Look for s_waitcnt lgkmcnt(0) before mfma → LDS latency exposed → apply Stage 2
grep -c "s_waitcnt lgkmcnt(0)" <path>.amdgcn
```

A high count of `s_waitcnt vmcnt(0)` indicates the kernel stalls waiting for
global DMA to complete — apply Stage 1. A high count of `s_waitcnt lgkmcnt(0)`
(after Stage 1, or instead of vmcnt stalls) indicates LDS read latency is exposed
— apply Stage 2.

---

## Background: Why Two Stages?

After applying async_copy with bank-conflict-free LDS, two independent latency
sources remain in a naive single-buffered kernel:

1. **Global memory DMA latency** — `wait_group(0)` stalls until DMA completes
   before any LDS reads begin. Visible as high `vmcnt` stall cycles in ATT traces.
2. **LDS read latency** — `ds_read` has ~20-40 cycle latency; if MFMA immediately
   follows, `lgkmcnt` stalls appear.

Stage 1 (global prefetch) hides latency (1). Stage 2 (local prefetch) then hides
latency (2). Together they achieve three-way overlap per iteration:

```
Time →
[DMA for k+2] ─────────────────────────────────────▶
              [ds_read for k+1] ──────────────▶
                                [MFMA for k] ────────▶
```

---

## Stage 1: Global Prefetch (Double Buffering)

### When to Apply

Use when ATT trace or amdgcn inspection shows:
- High-cycle `s_waitcnt vmcnt(0)` before MFMA
- VMEM category > 30% of total cycles
- MFMA utilization < 70%

If MFMA utilization already > 85%, skip — kernel is compute-bound.

### Pipeline Transformation

**Before (blocking single buffer):**
```
for k:
    DMA A[k], B[k] → LDS      # commit_group
    wait_group(0)              # STALL: wait for DMA to finish
    load from LDS
    MFMA
```

**After (overlapped double buffer):**
```
Prologue:
    DMA A[0], B[0] → buffer[0]   commit_group

for k in range(0, iterMax-1):
    DMA A[k+1], B[k+1] → buffer[1-k%2]   commit_group   ← new copy
    wait_group(1)                                         ← wait for PREV copy only
    load from buffer[k%2]                                 ← PREV copy's data
    MFMA(A[k], B[k])                                      ← runs while NEW copy fills buffer

Epilogue:
    wait_group(0)
    load from buffer[(iterMax-1)%2]
    MFMA(A[iterMax-1], B[iterMax-1])
```

`wait_group(1)` means "wait until at most 1 outstanding DMA group remains" — the
new copy can run concurrently with MFMA on the previous copy's data.

### Code Changes

#### 1. Allocate double-buffered shared memory

```python
# Before (single buffer):
smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

# After (double buffer — leading dimension = nBuffers):
nBuffers: gl.constexpr = 2
smemA = gl.allocate_shared_memory(
    a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA
)
smemB = gl.allocate_shared_memory(
    b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], sharedLayoutB
)
```

#### 2. Add prologue (before the loop)

```python
iterMax = gl.cdiv(K, BLOCK_K)

## Prologue: DMA tile 0 into buffer 0
g_idx = 0
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk
```

#### 3. Rewrite the loop (range changes, add buffer indexing, wait_group(1))

```python
for k in range(0, iterMax - 1):
    l_idx = k % 2       # buffer to READ from (prev DMA's data)
    g_idx = 1 - l_idx   # buffer to WRITE to (next DMA target)

    # Issue DMA for next tile (non-blocking)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    # Wait for PREVIOUS tile's DMA only (1 outstanding = the new one above)
    gl.amd.cdna4.async_copy.wait_group(1)

    # Read PREVIOUS tile from LDS and compute
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

#### 4. Add epilogue (after the loop)

```python
## Epilogue: drain final DMA and process last tile
gl.amd.cdna4.async_copy.wait_group(0)
l_idx = (iterMax - 1) % 2
a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
acc = gl.amd.cdna3.mfma(a, b, acc)
```

### Verify Correctness (Stage 1)

Write a self-contained test that imports both the old and new kernel from their
respective files and compares outputs numerically. Do not rely on local paths —
if the old kernel is inline or in a variable, copy it to a temp file first:

```python
import torch

# Run the reference kernel (single-buffer version)
def run_reference(a, b):
    # paste or import your single-buffer matmul here
    ...

# Run the Stage 1 kernel (double-buffer / global prefetch)
def run_stage1(a, b):
    # paste or import your stage-1 matmul here
    ...

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')

c_ref    = run_reference(a, b)
c_stage1 = run_stage1(a, b)
assert torch.allclose(c_ref, c_stage1, atol=1e-2, rtol=1e-2), \
    f"FAILED: max diff = {(c_ref - c_stage1).abs().max().item()}"
print("Stage 1 correctness OK, max diff:", (c_ref - c_stage1).abs().max().item())
```

### Measure Performance (Stage 1)

```bash
rocprofv3 --stats -f csv -- python3 <kernel.py> 2>&1 | grep -A5 "KernelName"
```

**Expected improvement:** 15–40% speedup depending on how memory-bound the original
was. Root cause: the DMA for tile k+1 now overlaps with MFMA on tile k, so global
memory latency is hidden inside compute time rather than adding to it.

**Check ATT trace after:** reduced `vmcnt` stall cycles; higher MFMA%; pattern
`DMA → MFMA (overlap) → wait_group(1) → ds_read → MFMA`.

---

## Stage 2: Local Prefetch (LDS Read Overlap)

### When to Apply

After Stage 1, run `/kernel-trace-analysis` on the updated kernel. Apply Stage 2 if:
- High-cycle `s_waitcnt lgkmcnt(0)` instructions remain
- `ds_read` instructions show significant stall counts
- Pattern visible: `ds_read ... → lgkmcnt stall → mfma`

If LDS stalls are < 10% of total cycle cost, Stage 2 won't help.

### Pipeline Transformation

**After Stage 1 (still has LDS stall):**
```
iter k:
    DMA[k+1] → buffer_g         (overlapped with MFMA above)
    wait_group(1)                (wait for DMA[k])
    ds_read A[k], B[k]           ← lgkmcnt stall here
    MFMA(A[k], B[k])
```

**After Stage 2 (three-way overlap):**
```
Prologue:
    DMA[0] → buffer[0]           commit_group
    DMA[1] → buffer[1]           commit_group
    wait_group(1)                (wait for DMA[0])
    ds_read A[0], B[0]           ← pre-load tile 0 into registers

iter k:
    MFMA(A[k], B[k])             ← A[k]/B[k] already in registers!
    wait_group(0)                (wait for DMA[k+1])
    DMA[k+2] → buffer_g          (new DMA, masked on last useful iter)
    ds_read A[k+1], B[k+1]       ← pre-load next iter's data into registers
    a = a_next; b = b_next       ← swap for next iteration

Epilogue:
    MFMA(a, b)                   ← final tile already in registers, no extra wait
```

### Code Changes

#### 1. Extend the prologue to issue TWO DMA groups and one LDS read

```python
iterMax = gl.cdiv(K, BLOCK_K)

## Prologue step 1: DMA tile 0 → buffer 0
g_idx = 0
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk

## Prologue step 2: DMA tile 1 → buffer 1
g_idx = 1
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk

## Wait for tile 0, then pre-load it from LDS into registers
gl.amd.cdna4.async_copy.wait_group(1)
l_idx = 0
a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
```

#### 2. Rewrite the loop: MFMA first, then wait + DMA + LDS read

```python
for k in range(0, iterMax - 1):
    g_idx = k % 2       # buffer that just finished → reuse for next DMA
    l_idx = 1 - g_idx   # buffer holding next iter's data

    ## MFMA on data already in registers (no stall — data was pre-loaded)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    ## Wait for inflight DMA (tile k+1, issued in prologue or previous iter)
    gl.amd.cdna4.async_copy.wait_group(0)

    ## Issue DMA for tile k+2 (masked on last useful iteration to avoid OOB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2))
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB.index(g_idx), b_base, b_offsets, mask=(k != (iterMax - 2))
    )
    gl.amd.cdna4.async_copy.commit_group()

    ## LDS read for next iteration (overlaps with new DMA above)
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    ## Swap: next iter's register data becomes current
    a = a_next
    b = b_next

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

#### 3. Simplify the epilogue

```python
## Epilogue: final MFMA — data already in registers from last LDS prefetch
acc = gl.amd.cdna3.mfma(a, b, acc)
# No extra wait/load needed
```

### Verify Correctness (Stage 2)

```python
import torch

# Run Stage 1 kernel as reference for Stage 2
def run_stage1(a, b):
    # paste or import your stage-1 matmul here
    ...

# Run Stage 2 kernel (local prefetch)
def run_stage2(a, b):
    # paste or import your stage-2 matmul here
    ...

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')

c_stage1 = run_stage1(a, b)
c_stage2 = run_stage2(a, b)
assert torch.allclose(c_stage1, c_stage2, atol=1e-2, rtol=1e-2), \
    f"FAILED: max diff = {(c_stage1 - c_stage2).abs().max().item()}"
print("Stage 2 correctness OK, max diff:", (c_stage1 - c_stage2).abs().max().item())
```

### Measure Performance (Stage 2)

```bash
rocprofv3 --stats -f csv -- python3 <kernel.py> 2>&1 | grep -A5 "KernelName"
```

**Expected improvement:** reduced `lgkmcnt` stall cycles; higher MFMA%. Root cause:
`ds_read` now runs one iteration ahead of the MFMA that needs the data. By the time
MFMA issues, the register file already holds the operands — no stall on the LDS
arbiter. The LDS read latency is hidden inside the previous MFMA's execution time.

**Important compiler note:** Full three-way overlap (DMA + ds_read + MFMA in parallel)
depends on the Triton/Gluon compiler scheduling the instructions correctly. If the
improvement is smaller than expected, the compiler may have serialized the schedule.
Check `arch_vgpr_count` from rocprofv3 — increased register pressure may also limit
occupancy.

---

## Performance Summary

| Stage | Hides | Mechanism | Typical Speedup |
|-------|-------|-----------|-----------------|
| Stage 1 (global prefetch) | DMA latency | `wait_group(1)` lets DMA run during MFMA | 15–40% |
| Stage 2 (local prefetch) | LDS read latency | `ds_read` issued one iter ahead; MFMA uses register data | 5–20% additional |

Speedup is higher when the original kernel has worse memory bottlenecks. A kernel
that is already close to peak MFMA utilization will gain less from pipelining.

---

## Fallback Guidance

**Global prefetch (Stage 1) shows no improvement:**
- Run `/kernel-trace-analysis` — kernel may be LDS-read-bound, not DMA-bound
- LDS stalls (lgkmcnt) dominate → proceed directly to Stage 2
- If kernel is compute-bound (MFMA > 85%), no memory optimization will help

**Local prefetch (Stage 2) shows no improvement:**
- Compiler may not schedule three-way overlap — check ISA output
- Kernel may be MFMA-bound after Stage 1
- Register pressure may reduce occupancy (`arch_vgpr_count` increased)
- Restore Stage 1 kernel and document

---

## Verification Checklist

After each stage:
- [ ] Correctness: `torch.allclose(c_ref, c_new, atol=1e-2, rtol=1e-2)` passes
- [ ] Performance: `rocprofv3 --stats` shows lower kernel duration
- [ ] ATT trace: target stall type (vmcnt / lgkmcnt) visibly reduced
- [ ] MFMA%: higher than before this stage
