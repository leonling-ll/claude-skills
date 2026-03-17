---
name: gluon-beyond-loop-opt
description: >
  Apply "beyond the hot-loop" optimizations to a Gluon GEMM kernel whose K-loop
  is already highly optimized (DMA/LDS latencies hidden, high MFMA utilization)
  but still has L2 cache inefficiency for matrix B or epilogue tail latency:
  (1) XCD-aware PID remapping — a get_pids() helper that spreads consecutive
  program IDs across XCDs (compute dies) to improve L2 cache reuse for matrix B,
  combined with grouped swizzle (GROUP_SIZE_M) to reuse B tiles across M-blocks;
  (2) epilogue M-slicing — break the final MFMA accumulator into 4 slices of 64
  rows using extract_slice, interleaving each slice's MFMA with a buffer_store of
  the previous slice's results to hide store latency and reduce register pressure
  spike. Apply when rocprofv3 shows low L2 cache hit rate for reads, or when the
  amdgcn shows high VGPR count and s_nop/vmcnt stalls in the epilogue. Most
  effective on multi-XCD GPUs (MI300X: 8 XCDs, MI350: varies). 
  Usage: /gluon-beyond-loop-opt
---

# Gluon GEMM: Beyond-Loop Optimizations (XCD Remap + Epilogue Slicing)

Apply two "beyond the K-loop" optimizations to a Gluon GEMM kernel whose main
loop is already well-optimized (DMA and LDS latencies hidden, high MFMA
utilization), but still shows:
- L2 cache misses for matrix B reads (rocprofv3 shows low L2 hit rate)
- Register pressure spikes in the epilogue (high VGPR count in amdgcn)
- Buffer store latency stalls at the end of the kernel (s_nop / vmcnt in epilogue)

## Background

Two distinct bottlenecks remain after the K-loop is fully pipelined:

1. **XCD-aware PID mapping**: On MI300X (8 XCDs), the default flat PID mapping
   fills one XCD before starting the next. Consecutive matrix B tiles (same column)
   cannot be reused in L2 across XCDs. XCD remapping spreads work so that
   each XCD covers a wider range of the output matrix, improving L2 hit rate for B.

2. **Epilogue M-slicing**: After the final K-iteration's MFMA, the full 256×128
   accumulator is live. Converting it and storing causes a register-pressure spike.
   Breaking it into 4 × 64-row slices via `extract_slice` lets us issue MFMA slice N+1
   while storing slice N, hiding the store latency.

## Step 0: Check GPU Platform

```bash
python3 -c "import torch; props = torch.cuda.get_device_properties(0); print(props.gcnArchName, props.multi_processor_count)"
```

- **XCD remapping**: Most beneficial on MI300X (gfx942, 8 XCDs) and MI350 (gfx950).
  `NUM_XCDS = 8` for MI300X/MI308X; `NUM_XCDS = 1` disables remapping.
- **Epilogue slicing**: Applies to both gfx942 and gfx950.

## Step 0b: Assess Register Pressure from amdgcn

```bash
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5
# Check VGPR count in the kernel header
grep "num_vgprs\|; \.vgpr_count\|VGPR" <path>.amdgcn | head -5
# Also look for s_nop in epilogue (indicates store-compute serialization)
grep -c "s_nop\|s_waitcnt vmcnt" <path>.amdgcn
```

High VGPR count (e.g., > 200) and frequent `s_nop` / `s_waitcnt vmcnt` in the
epilogue confirm that epilogue slicing will help. Low L2 hit rate in rocprofv3
output confirms that XCD remapping and grouped swizzle will help.

## Stage 1: XCD-Aware PID Remapping + Grouped Swizzle

### When to Apply

Apply Stage 1 when rocprofv3 shows low L2 cache hit rate for matrix B reads,
especially on large matrices (M, N >= 4096) on MI300X or MI350.

### Performance Gain Reasoning

**Default flat mapping on 8 XCDs:**
```
XCD0: pid 0,1,2,...,N/8-1     → rows 0..BLOCK_M*(N/8-1)
XCD1: pid N/8,...,2N/8-1      → next rows
...
```
Each XCD has a contiguous row range — different XCDs compute different B columns —
no L2 sharing across XCDs. Adjacent PIDs on the same XCD access the same B tile,
but PIDs on different XCDs never share B tiles.

**With XCD remapping (stride by NUM_XCDS):**
```
XCD0: pid 0, 8, 16, ...       → scattered rows across all N
XCD1: pid 1, 9, 17, ...       → same column tiles as XCD0 for some rows
...
```
Adjacent PIDs (XCD0's pid 0 and XCD1's pid 1) hit the same B column tile —
L2 sharing across XCDs. This improves L2 hit rate for B reads significantly.

**Grouped swizzle (GROUP_SIZE_M):** GROUP_SIZE_M M-tiles share the same N-column
work — B tiles are reused in L2 across M-blocks. Within a group, all tiles read the
same set of B columns, so B tiles stay in L2 across the group.

### Step 1: Add the XCD-Aware PID Remapping Helper

Create a separate `get_pids` gluon function:

```python
from triton.experimental.gluon.language.amd.cdna3 import extract_slice

@gluon.jit
def get_pids(
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        ## XCD remapping: spread consecutive PIDs across XCDs
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        ## Simple row-major mapping
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        ## Grouped swizzle: GROUP_SIZE_M rows share the same N tiles (B reuse)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n
```

### Step 2: Add New Kernel Parameters

Add `GRID_MN`, `NUM_XCDS`, `GROUP_SIZE_M` as `gl.constexpr` parameters:

```python
@gluon.jit
def gemm_kernel(
    ...
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)
    # ... rest of kernel (same as before, replacing old pid_m/pid_n computation)
```

Replace the original `pid = gl.program_id(0); pid_m = pid // ...; pid_n = pid % ...`
with the `get_pids` call.

### Step 3: Update the Kernel Launch Site

```python
def matmul(a, b, c=None):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 4
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8       # 8 for MI300X/MI308X; 1 to disable
    GROUP_SIZE_M = 4   # group 4 M-tiles to share B tiles in L2

    gemm_kernel[grid](
        ...
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
```

### Stage 1 Correctness Check

```python
import torch

# Inline reference: simple matmul using torch for correctness baseline
def ref_matmul(a, b):
    return torch.matmul(a.float(), b.float()).half()

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')

c_ref = ref_matmul(a, b)
c_stage1 = matmul(a, b)  # your kernel with get_pids applied
assert torch.allclose(c_ref, c_stage1, atol=1e-2, rtol=1e-2), "FAILED"
print("Stage 1 correctness OK, max diff:", (c_ref - c_stage1).abs().max().item())
```

### Stage 1 Performance Measurement (Mode 2 — counter collection)

Use `/kernel-perf-analysis` in **Mode 2** to measure L2 cache hit rate for matrix B
reads before and after XCD remapping. Mode 2 is triggered by mentioning "counter",
"cache hit", or a hardware counter name:

**Before XCD remapping:**
```
/kernel-perf-analysis
Kernel file: <absolute path to baseline_kernel.py>
Mode hint: counter cache hit TCP_TCC_READ_REQ_sum TCC_EA0_RDREQ_DRAM_sum
Label: before_xcd_remap
```

**After XCD remapping:**
```
/kernel-perf-analysis
Kernel file: <absolute path to stage1_kernel.py>
Mode hint: counter cache hit TCP_TCC_READ_REQ_sum TCC_EA0_RDREQ_DRAM_sum
Label: after_xcd_remap
```

Expected output showing improved L2 utilization (fewer DRAM requests per L2 request):
```
| Version         | TCP_TCC_READ_REQ_sum | TCC_EA0_RDREQ_DRAM_sum | Dispatches |
|-----------------|---------------------|------------------------|------------|
| before_xcd_remap|           1,024,512 |                512,000 |         20 |
| after_xcd_remap |           1,024,512 |                180,000 |         20 |
```

Lower `TCC_EA0_RDREQ_DRAM_sum` with unchanged `TCP_TCC_READ_REQ_sum` means more B
tile reads are served from L2 — confirming the XCD remapping is effective.

**Expected gain**: 3-8% throughput improvement from better L2 utilization, more
pronounced on MI300X with large matrices where B tiles are large relative to L2 per XCD.

## Stage 2: Epilogue M-Slicing

### When to Apply

Apply Stage 2 when the amdgcn shows:
- High VGPR count in the kernel header (> 200 VGPRs)
- Frequent `s_nop` or `s_waitcnt vmcnt` instructions in the epilogue section

These indicate that the 256×128 accumulator is live all at once during the epilogue,
causing register pressure and serializing MFMA with buffer_store.

### Performance Gain Reasoning

The 256×128 FP32 accumulator held live requires a large number of VGPRs. When all
accumulator values are live simultaneously during type conversion and store, the
register file is saturated — the compiler inserts `s_nop` bubbles to serialize
MFMA and store operations.

Slicing the epilogue into 4 × 64-row chunks via `extract_slice`:
- Only 64×128 values are live at once per slice (4x fewer registers)
- MFMA for slice N+1 can issue while slice N's store is in flight
- Store latency (buffer_store is async) is hidden behind the next MFMA
- Register pressure spike is eliminated — compiler can reduce VGPR allocation

### Step 4: Add Epilogue Slicing with extract_slice

Import the slice helper:
```python
from triton.experimental.gluon.language.amd.cdna3 import extract_slice
```

In the epilogue (last MFMA region, iterMax-1), break the 256-row MFMA into
4 × 64-row slices. For the left accumulator (example):

```python
## Epilogue: Region 2 (left half, iterMax-1)
# Precompute slice base addresses
offs_cm_slice = gl.arange(0, BLOCK_M // 4, gl.SliceLayout(1, gStoreLayoutC))
c_slice_offsets = stride_cm * offs_cm_slice[:, None] + stride_cn * offs_cn[None, :]
c00_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
c01_base = c00_base + 64 * stride_cm   # row 64
c02_base = c01_base + 64 * stride_cm   # row 128
c03_base = c02_base + 64 * stride_cm   # row 192

## Slice 0: A[0:64,:] × B_left → acc_left[0:64,:]
a0   = extract_slice(a, [64, 64], [0, 0])
acc00 = extract_slice(acc_left, [64, 128], [0, 0])
acc00 = gl.amd.cdna3.mfma(a0, b_left, acc00)

## Wait for last DMA, load B_right
gl.amd.cdna4.async_copy.wait_group(0)
b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

## Slice 1: A[64:128,:] × B_left → acc_left[64:128,:]
a1   = extract_slice(a, [64, 64], [64, 0])
acc01 = extract_slice(acc_left, [64, 128], [64, 0])
acc01 = gl.amd.cdna3.mfma(a1, b_left, acc01)

## Store slice 0 (while slice 1 MFMA runs)
c00 = acc00.to(a_ptr.dtype.element_ty)
c00 = gl.convert_layout(c00, layout=gStoreLayoutC)
gl.amd.cdna3.buffer_store(stored_value=c00, ptr=c00_base, offsets=c_slice_offsets)

## Slice 2, store slice 1, slice 3, store slice 2 ...
## (repeat pattern for a2/acc02, a3/acc03)
```

The key pattern: **issue MFMA for slice N+1, then store slice N**. This creates
a pipeline where MFMA and store overlap.

The full epilogue applies this pattern for all 8 slices (4 for left half, 4 for right half
of the accumulator).

### Stage 2 Correctness Check

```python
import torch

# Inline reference: simple matmul using torch for correctness baseline
def ref_matmul(a, b):
    return torch.matmul(a.float(), b.float()).half()

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')

c_ref = ref_matmul(a, b)
c_stage2 = matmul(a, b)  # your kernel with epilogue slicing applied
assert torch.allclose(c_ref, c_stage2, atol=1e-2, rtol=1e-2), "FAILED"
print("Stage 2 correctness OK, max diff:", (c_ref - c_stage2).abs().max().item())
```

### Stage 2 Performance Measurement (Mode 1 — perf table)

Use `/kernel-perf-analysis` in **Mode 1** to measure the TFLOPS and VGPR count
improvement from epilogue slicing. Mode 1 is triggered by mentioning "TFLOPS",
"perf table", "VGPR", or "benchmark":

```
/kernel-perf-analysis
Kernel file: <absolute path to stage2_kernel.py>
Mode hint: perf table TFLOPS VGPR benchmark
Label: stage2_epilogue_slice
```

Expected output showing reduced VGPRs and improved TFLOPS:
```
| Version               | TFLOPS | VGPRs | Spills | MFMA Eff. | avg time  |
|-----------------------|--------|-------|--------|-----------|-----------|
| stage2_epilogue_slice |    152 |   176 |      0 |    92.1%  | 622.4 us  |
```

Also confirm ISA improvements (fewer s_nop and smaller VGPR count):
```bash
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5
grep "num_vgprs\|; \.vgpr_count\|VGPR" <path>.amdgcn | head -5
grep -c "s_nop\|s_waitcnt vmcnt" <path>.amdgcn
```

Compare VGPR count and `s_nop` count before and after Stage 2. A successful
application will show reduced VGPR count and fewer `s_nop` instructions in the
epilogue section of the amdgcn.

**Expected gain**: 2-7% throughput improvement from reduced register pressure and
hidden store latency. Combined with Stage 1, total improvement is typically 5-15%
over a fully pipelined K-loop kernel.

## Tuning Parameters

- **`NUM_XCDS`**: Set to actual XCD count of the GPU.
  - MI300X / MI308X: `NUM_XCDS = 8`
  - MI350: check with the Step 0 platform check
  - Use `NUM_XCDS = 1` to disable XCD remapping for A/B testing Stage 1 in isolation.
- **`GROUP_SIZE_M`**: Controls how many M-tiles share B tiles in L2.
  Typical values: 4, 8. Tune based on matrix shape and L2 cache size per XCD.
- **`BLOCK_M // 4 = 64`**: The slice size is fixed at 64 rows (4 slices of 64 = 256 rows).

## Fallback

If no improvement:
1. **XCD remapping only**: Try `GROUP_SIZE_M=1` (no swizzle) first to isolate effects
2. **Epilogue slicing only**: Apply slicing without XCD remapping to isolate effects
3. **Matrix shape matters**: XCD remapping benefits are more pronounced for large M,N
4. Restore the original kernel if neither optimization helps

## XCD Remapping Intuition

Default flat mapping on 8 XCDs:
```
XCD0: pid 0,1,2,...,N/8-1     → rows 0..BLOCK_M*(N/8-1)
XCD1: pid N/8,...,2N/8-1      → next rows
...
```
Each XCD has a contiguous row range → different XCDs compute different B columns → no L2 sharing.

With XCD remapping (stride by NUM_XCDS):
```
XCD0: pid 0, 8, 16, ...       → scattered rows across all N
XCD1: pid 1, 9, 17, ...       → same column tiles as XCD0 for some rows
...
```
Adjacent PIDs (XCD0's pid 0 and XCD1's pid 1) hit the same B column tile → L2 sharing across XCDs.
