---
name: gluon-pipeline-opt
description: >
  Apply pipeline optimizations to a Gluon GEMM kernel to hide global memory latency
  (vmcnt stalls) and LDS read latency (lgkmcnt stalls) on both CDNA3 (gfx942/MI300X)
  and CDNA4 (gfx950/MI350). Two progressive stages: Stage 1 (global prefetch / double
  buffering) — on CDNA4 uses async_copy DMA with wait_group(1); on CDNA3 uses
  buffer_load→VGPR staging→ds_write with s_waitcnt vmcnt(0), holding two full tiles
  in VGPRs simultaneously. Stage 2 (local prefetch) — architecture-independent,
  issues ds_read one iteration ahead of MFMA so LDS read latency is hidden behind
  compute; applies to any CDNA generation. Apply Stage 1 when ATT trace shows vmcnt
  stalls before ds_write (CDNA3) or MFMA (CDNA4). Apply Stage 2 when lgkmcnt stalls
  remain. CDNA3 requires monitoring VGPR occupancy — double tile buffering in VGPRs
  may reduce occupancy and negate gains. Trigger for global prefetch, double buffering,
  local prefetch, vmcnt stalls, lgkmcnt stalls, ds_read overlap, or hiding memory
  latency in Gluon GEMM kernels on MI300X, MI308X, MI325X, or MI350.
  Usage: /gluon-pipeline-opt
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Gluon Pipeline Optimization: Global + Local Prefetch

Two progressive pipeline stages that overlap global memory loads, LDS reads, and MFMA
compute to hide memory latency. Apply Stage 1 first; verify it is effective before
proceeding to Stage 2.

**Both stages support CDNA3 and CDNA4.** Each stage has one unified code template —
the only arch-specific difference is the global load mechanism (one line swap).

---

## Step 0: Identify GPU Platform and Baseline Bottlenecks

```bash
# Identify architecture
python3 -c "import torch; props = torch.cuda.get_device_properties(0); print(props.gcnArchName)"

# Find most recent compiled kernel ISA
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5

# Count stall signals in the hot loop
grep -c "s_waitcnt vmcnt(0)"   <path>.amdgcn   # Stage 1 signal: HBM/DMA latency exposed
grep -c "s_waitcnt lgkmcnt(0)" <path>.amdgcn   # Stage 2 signal: LDS read latency exposed
```

| `gcnArchName` | Architecture | `async_copy` | Global load mechanism              |
|---------------|--------------|--------------|-------------------------------------|
| `gfx942`      | CDNA3        | No           | `buffer_load` → VGPR → `ds_write`  |
| `gfx950`      | CDNA4        | Yes          | `async_copy.buffer_load_to_shared`  |

MI300X, MI308X, and MI325X are all `gfx942` (CDNA3). MI350 is `gfx950` (CDNA4).

| Signal in amdgcn                                  | Root cause             | Fix     |
|---------------------------------------------------|------------------------|---------|
| `s_waitcnt vmcnt(0)` before `ds_write` (CDNA3)   | HBM load latency       | Stage 1 |
| `s_waitcnt vmcnt(0)` before MFMA (CDNA4)         | DMA latency            | Stage 1 |
| `s_waitcnt lgkmcnt(0)` before MFMA               | LDS read latency       | Stage 2 |

**If MFMA utilization is already > 85%, the kernel is compute-bound — skip both stages.**

---

## Background: Two Latency Sources

After applying bank-conflict-free LDS layouts, two independent latency sources remain:

1. **Global memory latency** (~200–800 cycles): HBM → VGPR (CDNA3) or HBM → LDS via DMA (CDNA4).
2. **LDS read latency** (~40–100 cycles): `ds_read` issues to VGPR registers.

Stage 1 hides (1). Stage 2 hides (2). When both are applied:

```
Time →  [global load for k+2] ────────────────────────────────────────▶
                               [ds_read for k+1] ──────────────▶
                                                 [MFMA for k] ──────────▶
```

### VGPR Cost (CDNA3 Only)

On CDNA3, Stage 1 holds two full tiles in VGPRs simultaneously. This can reduce
occupancy. After Stage 1, check:

```bash
# Inspect the compiled ISA
grep "NumVgprs:" <path>.amdgcn
# occupancy = floor(512 / ceil(NumVgprs / 8) / 8)  [gfx942 has 512 VGPRs/SIMD, granularity 8]
```

If VGPRs increased so much that occupancy dropped from 2→1 waves/SIMD and the
kernel is slower, revert Stage 1 and document. Stage 2 can still be attempted
if lgkmcnt stalls dominate without Stage 1 in place.

---

## Stage 1: Global Prefetch (Double Buffering)

### What It Does

Allocate two LDS buffers and pipeline the global load for tile `k+1` to run
concurrently with the MFMA for tile `k`. The key difference from the baseline:

- **Before:** load tile k → wait → write to LDS → MFMA (load blocks MFMA)
- **After:** load tile k+1 in background → MFMA tile k → write tile k+1 to LDS
  (global load overlaps with MFMA)

### Code Template

The template is identical for CDNA3 and CDNA4. Only the two marked lines differ.

#### 1. Allocate double-buffered LDS

```python
nBuffers: gl.constexpr = 2
smemA = gl.allocate_shared_memory(
    a_ptr.type.element_ty, [nBuffers, BLOCK_M, BLOCK_K], layout=sharedLayoutA
)
smemB = gl.allocate_shared_memory(
    b_ptr.type.element_ty, [nBuffers, BLOCK_K, BLOCK_N], layout=sharedLayoutB
)
```

#### 2. Prologue — load tile 0 into LDS[0]

```python
iterMax = gl.cdiv(K, BLOCK_K)
gl.assume(iterMax > 0)

g_idx = 0

# ── CDNA4 ──────────────────────────────────────────────────────────────────
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
# ── CDNA3 (replace the three lines above with these four) ──────────────────
vgpr_a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=a_offsets)
vgpr_b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=b_offsets)
smemA.index(g_idx).store(vgpr_a)    # ds_write; s_waitcnt vmcnt(0) fires here
smemB.index(g_idx).store(vgpr_b)
# ───────────────────────────────────────────────────────────────────────────

a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk
```

#### 3. Main loop — overlap load k+1 with MFMA k

```python
for k in range(0, iterMax - 1):
    l_idx = k % 2        # LDS slot holding the tile to compute NOW
    g_idx = 1 - l_idx    # LDS slot to load the NEXT tile into

    # Issue global load for tile k+1 (non-blocking)
    # ── CDNA4 ──────────────────────────────────────────────────────────────
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(1)   # allow 1 DMA in-flight while we compute
    # ── CDNA3 (replace the four lines above with these two) ────────────────
    vgpr_a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=a_offsets)
    vgpr_b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=b_offsets)
    # ───────────────────────────────────────────────────────────────────────

    # LDS read + MFMA for tile k (overlaps with in-flight global load above)
    a = smemA.index(l_idx).load(layout=dotOpLayoutA)
    b = smemB.index(l_idx).load(layout=dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    # Write tile k+1 into LDS (vmcnt stall hidden behind MFMA above)
    # ── CDNA4: no ds_write needed — async_copy already landed in LDS ───────
    # ── CDNA3 (add these two lines after MFMA) ─────────────────────────────
    smemA.index(g_idx).store(vgpr_a)
    smemB.index(g_idx).store(vgpr_b)
    # ───────────────────────────────────────────────────────────────────────

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

#### 4. Epilogue — drain remaining in-flight load and compute last tile

```python
# ── CDNA4 ──────────────────────────────────────────────────────────────────
gl.amd.cdna4.async_copy.wait_group(0)
# ── CDNA3 (no extra wait needed — vmcnt(0) already fired in last loop iter) -
# ───────────────────────────────────────────────────────────────────────────

l_idx = (iterMax - 1) % 2
a = smemA.index(l_idx).load(layout=dotOpLayoutA)
b = smemB.index(l_idx).load(layout=dotOpLayoutB)
acc = gl.amd.cdna3.mfma(a, b, acc)
```

---

### Stage 1 Verification ✓

Run this after implementing Stage 1. Do not proceed to Stage 2 unless Stage 1
passes both checks.

#### Correctness

```python
import torch, importlib.util

def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

baseline = load_kernel("kernel_baseline.py", "baseline")
stage1   = load_kernel("kernel_stage1.py",   "stage1")

# Use the actual input shapes from the target workload
x = torch.randn(B, M, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

c_ref = baseline.launcher(x, w)
c_new = stage1.launcher(x, w)
assert torch.allclose(c_ref, c_new, atol=1.0, rtol=0), \
    f"Stage 1 FAILED: max diff = {(c_ref - c_new).abs().max().item()}"
print("Stage 1 correctness OK")
```

#### Performance + ISA check

```python
# Warmup, then time both
for _ in range(10): baseline.launcher(x, w); stage1.launcher(x, w)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(200): baseline.launcher(x, w)
end.record(); torch.cuda.synchronize()
base_us = start.elapsed_time(end) / 200 * 1000

start.record()
for _ in range(200): stage1.launcher(x, w)
end.record(); torch.cuda.synchronize()
s1_us = start.elapsed_time(end) / 200 * 1000

print(f"Baseline:  {base_us:.1f} µs")
print(f"Stage 1:   {s1_us:.1f} µs  ({base_us/s1_us:.3f}x)")
```

```bash
# ISA: confirm vmcnt stall count dropped and VGPR occupancy is acceptable
stage1_isa=$(find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -1 | awk '{print $NF}')
echo "VGPRs:"; grep "NumVgprs:" $stage1_isa
echo "vmcnt(0) count:";   grep -c "s_waitcnt vmcnt(0)"   $stage1_isa
echo "lgkmcnt(0) count:"; grep -c "s_waitcnt lgkmcnt(0)" $stage1_isa
```

#### Decision

| Outcome | Action |
|---------|--------|
| Faster **and** `vmcnt(0)` count dropped | ✅ Stage 1 succeeded — proceed to Stage 2 |
| Slower, CDNA3, VGPRs increased significantly | ❌ Revert. Check if occupancy dropped (waves/SIMD fell). Document and stop. |
| Slower, CDNA4 | ❌ Revert. Kernel is likely compute-bound or already has good HW prefetch. Stop. |
| Same speed, `lgkmcnt(0)` count is high | ⚠️ Stage 1 neutral — LDS latency dominates. Proceed to Stage 2 anyway. |

---

## Stage 2: Local Prefetch (LDS Read Overlap)

**Architecture-independent.** This stage is purely a register scheduling technique —
no `async_copy` required.

### What It Does

After Stage 1, `ds_read` may still stall before MFMA. Fix: issue `ds_read` for tile
`k+1` at the **end** of iteration `k`, so by the time iteration `k+1` reaches MFMA
the data is already in registers.

- **Before (after Stage 1):** `wait(1)` → `ds_read k` → lgkmcnt stall → `MFMA k`
- **After:** `MFMA k` uses registers pre-loaded at end of iteration `k-1` — no stall

This requires extending the prologue to load **two** tiles and pre-read the first
into registers before the loop begins.

### Code Template

The template is identical for CDNA3 and CDNA4. Only the two marked lines differ
(same as Stage 1).

#### 1. Extended prologue — load tiles 0 and 1, pre-read tile 0 into registers

```python
iterMax = gl.cdiv(K, BLOCK_K)
gl.assume(iterMax > 1)

## --- Tile 0 → LDS[0] ---
g_idx = 0
# ── CDNA4 ──────────────────────────────────────────────────────────────────
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
# ── CDNA3 ──────────────────────────────────────────────────────────────────
vgpr_a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=a_offsets)
vgpr_b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=b_offsets)
smemA.index(g_idx).store(vgpr_a)
smemB.index(g_idx).store(vgpr_b)
# ───────────────────────────────────────────────────────────────────────────
a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk

## --- Tile 1 → LDS[1] ---
g_idx = 1
# ── CDNA4 ──────────────────────────────────────────────────────────────────
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
gl.amd.cdna4.async_copy.wait_group(1)   # wait for tile 0 only; tile 1 still in-flight
# ── CDNA3 ──────────────────────────────────────────────────────────────────
vgpr_a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=a_offsets)
vgpr_b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=b_offsets)
smemA.index(g_idx).store(vgpr_a)    # vmcnt stall fires here for tile 1
smemB.index(g_idx).store(vgpr_b)
# ───────────────────────────────────────────────────────────────────────────
a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk

## --- Pre-read tile 0 from LDS[0] into registers ---
# (tile 0 is guaranteed ready; tile 1 DMA/vmcnt may still be in-flight — that's fine)
a = smemA.index(0).load(layout=dotOpLayoutA)
b = smemB.index(0).load(layout=dotOpLayoutB)
```

#### 2. Main loop — MFMA first, then load k+2, then ds_read k+1

```python
for k in range(0, iterMax - 1):
    g_idx = k % 2        # LDS slot to write tile k+2 into (was consumed at iter k-1)
    l_idx = 1 - g_idx    # LDS slot holding tile k+1 (ready to read)

    ## MFMA on pre-loaded registers — no lgkmcnt stall
    acc = gl.amd.cdna3.mfma(a, b, acc)

    ## Issue global load for tile k+2 (non-blocking; masked on last useful iter)
    if k < iterMax - 2:
        # ── CDNA4 ──────────────────────────────────────────────────────────
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)   # drain — tile k+1 must be ready for ds_read below
        # ── CDNA3 ──────────────────────────────────────────────────────────
        vgpr_a = gl.amd.cdna3.buffer_load(ptr=a_ptr, offsets=a_offsets)
        vgpr_b = gl.amd.cdna3.buffer_load(ptr=b_ptr, offsets=b_offsets)
        # ───────────────────────────────────────────────────────────────────
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## Write tile k+2 into LDS (vmcnt stall hidden behind MFMA above)
    if k < iterMax - 2:
        # ── CDNA3 only ─────────────────────────────────────────────────────
        smemA.index(g_idx).store(vgpr_a)
        smemB.index(g_idx).store(vgpr_b)
        # ── CDNA4: async_copy already landed tile k+2 in LDS — nothing to do

    ## ds_read tile k+1 into registers for next MFMA (overlaps with ds_write above)
    a = smemA.index(l_idx).load(layout=dotOpLayoutA)
    b = smemB.index(l_idx).load(layout=dotOpLayoutB)
```

#### 3. Epilogue — final MFMA, data already in registers

```python
## Final MFMA — a, b were pre-loaded at the end of the last loop iteration
acc = gl.amd.cdna3.mfma(a, b, acc)
```

---

### Stage 2 Verification ✓

Run this after implementing Stage 2. Compare against the Stage 1 kernel (not the
original baseline) to isolate Stage 2's contribution.

#### Correctness

```python
c_s1  = stage1.launcher(x, w)
c_s2  = stage2.launcher(x, w)
assert torch.allclose(c_s1, c_s2, atol=1.0, rtol=0), \
    f"Stage 2 FAILED: max diff = {(c_s1 - c_s2).abs().max().item()}"
print("Stage 2 correctness OK")
```

#### Performance + ISA check

```python
# Same timing harness as Stage 1 — compare stage1 vs stage2
start.record()
for _ in range(200): stage1.launcher(x, w)
end.record(); torch.cuda.synchronize()
s1_us = start.elapsed_time(end) / 200 * 1000

start.record()
for _ in range(200): stage2.launcher(x, w)
end.record(); torch.cuda.synchronize()
s2_us = start.elapsed_time(end) / 200 * 1000

print(f"Stage 1:  {s1_us:.1f} µs")
print(f"Stage 2:  {s2_us:.1f} µs  ({s1_us/s2_us:.3f}x over Stage 1)")
```

```bash
stage2_isa=$(find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -1 | awk '{print $NF}')
echo "VGPRs:"; grep "NumVgprs:" $stage2_isa
echo "vmcnt(0) count:";   grep -c "s_waitcnt vmcnt(0)"   $stage2_isa
echo "lgkmcnt(0) count:"; grep -c "s_waitcnt lgkmcnt(0)" $stage2_isa
# Confirm lgkmcnt(0) count dropped compared to Stage 1
```

#### Decision

| Outcome | Action |
|---------|--------|
| Faster **and** `lgkmcnt(0)` count dropped | ✅ Stage 2 succeeded — keep it |
| Slower or same speed | ❌ Revert to Stage 1. Compiler already schedules ds_read well, or kernel is MFMA-bound. Document. |
| VGPRs increased, CDNA3 only | Check occupancy. If waves/SIMD dropped, revert. |

---

## Performance Summary

| Stage | Hides | Mechanism | Expected Speedup |
|-------|-------|-----------|------------------|
| Stage 1 (CDNA4) | DMA latency (~200–800 cy) | `wait_group(1)` overlaps DMA with MFMA | 15–40% |
| Stage 1 (CDNA3) | HBM latency (~200–800 cy) | `buffer_load` into VGPR, vmcnt hidden behind MFMA | 5–25% (VGPR-dependent) |
| Stage 2 (both)  | LDS read latency (~40–100 cy) | `ds_read` one iter ahead; MFMA uses pre-loaded registers | 5–20% additional |
