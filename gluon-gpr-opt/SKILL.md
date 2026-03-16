---
name: gluon-gpr-opt
description: >
  Apply general-purpose register (GPR) reduction and pipeline interleaving
  optimizations to a Gluon GEMM kernel with double-buffered LDS and local prefetch:
  (1) loop unroll ×2 — eliminate k%2 modulo overhead by making K a constexpr and
  duplicating the loop body with hardcoded buffer indices (g_idx=0/1, l_idx=1/0),
  reducing SALU instructions and enabling static index resolution; (2) N-slice —
  split the B tile into B_left and B_right half-width sub-tiles with separate
  async copy groups and accumulators, using wait_group(2) to allow B_right LDS
  reads to overlap B_left MFMA. Apply when: ISA shows s_and/v_and from k%2 modulo,
  or ATT trace shows residual pipeline bubbles after DMA and LDS latencies are
  hidden. Requires CDNA4 (gfx950/MI350). Trigger for /gemm-v6-loop-unroll and
  /gemm-v7-n-slice requests.
  Usage: /gluon-gpr-opt
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Gluon GEMM: GPR Reduction and Pipeline Interleaving Optimizations

This skill covers two complementary optimizations for a Gluon GEMM kernel that
already has double-buffered LDS and async copy in place:

- **Stage 1 — Loop Unroll ×2**: eliminate the `k % 2` modulo overhead that
  compiles to `s_and_b32`/`v_and_b32` instructions in the inner loop ISA.
- **Stage 2 — N-Slice**: split the B tile into left/right halves to create
  finer-grained async copy groups and additional MFMA/DMA overlap.

Apply Stage 1 first; apply Stage 2 only if profiling shows residual pipeline
bubbles after Stage 1 is in place.

---

## Step 0: Check Platform

Both optimizations require CDNA4 (gfx950 / MI350) for async_copy support.

```bash
python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
# Expected: gfx950
```

If the output is not `gfx950`, stop. The `gl.amd.cdna4.async_copy` API is not
available on earlier architectures.

---

# Stage 1: Loop Unroll ×2

## When to Apply

Compile the kernel and inspect the hot-loop ISA for modulo instructions:

```bash
# Find the compiled kernel ISA in Triton cache
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5
# Then inspect the hot loop for s_and/v_and instructions (k%2 overhead)
grep -c "s_and_b32\|v_and_b32" <path>.amdgcn
```

Apply Stage 1 when either of these is true:

1. **ISA evidence**: the `.amdgcn` file contains `s_and_b32` or `v_and_b32`
   instructions inside the main loop body, indicating the compiler emitted a
   runtime modulo for `k % 2`.
2. **ATT trace evidence**: an ATT wavefront trace shows SALU-heavy patterns
   (high SALU:VALU ratio) or unexplained stall cycles between consecutive MFMA
   dispatches that are not explained by LDS or DMA latency.

## Background

In a double-buffered kernel with local prefetch, the loop alternates between two
LDS buffer slots using:

```python
g_idx = k % 2
l_idx = 1 - g_idx
```

While correct, this modulo introduces:

- **Extra SALU instructions** every iteration to compute `k % 2`
- **Dynamic index computation** that prevents the compiler from statically
  resolving `smemA.index(g_idx)` and `smemA.index(l_idx)` at compile time
- **Potential compiler pessimism** around loop-carried state, reducing ILP

By unrolling the loop by 2, each even/odd iteration becomes a hardcoded copy
with `g_idx=0, l_idx=1` (first half) and `g_idx=1, l_idx=0` (second half). The
compiler can statically resolve all buffer accesses, eliminating the SALU
overhead and enabling better instruction scheduling.

Making `K` a `gl.constexpr` is also required so that `iterMax` is a compile-time
constant, which is a prerequisite for the unrolled range.

## Performance Gain Explanation

Loop unroll eliminates the runtime modulo computation (`k % 2 → s_and_b32`)
from every iteration. With `g_idx` and `l_idx` hardcoded as literals, the
compiler resolves `smemA.index(0)` and `smemA.index(1)` statically, removing
the corresponding SALU address computation. This reduces the SALU pressure per
loop iteration, freeing the SALU pipeline for other address calculations and
improving instruction-level parallelism (ILP) between MFMA and DMA.

## Step 1.1: Make K a constexpr

**Before:**
```python
def kernel(
    ...
    K,           # runtime value
    ...
):
    iterMax = gl.cdiv(K, BLOCK_K)
```

**After:**
```python
def kernel(
    ...
    K: gl.constexpr,   # compile-time constant
    ...
):
    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 3)   # needed for safe unrolling; at least 2 full unrolled iterations
```

Update the kernel call site — Triton propagates constexpr arguments automatically
when the parameter is declared with `gl.constexpr`:

```python
kernel[grid](
    ...
    K,
    ...
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    num_warps=num_warps,
)
```

## Step 1.2: Change the Loop Step to 2

**Before:**
```python
for k in range(0, iterMax - 1):
```

**After:**
```python
for k in range(0, iterMax - 2, 2):   # step by 2; epilogue handles last 2 iters
```

## Step 1.3: Duplicate the Loop Body with Hardcoded Indices

**Before (single body, runtime indices):**
```python
for k in range(0, iterMax - 1):
    g_idx = k % 2
    l_idx = 1 - g_idx

    acc = gl.amd.cdna3.mfma(a, b, acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), ...)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), ...)
    gl.amd.cdna4.async_copy.commit_group()
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
    a = a_next
    b = b_next
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

**After (unrolled with static indices):**
```python
for k in range(0, iterMax - 2, 2):
    ## --- First half: g_idx=0, l_idx=1 ---
    g_idx = 0
    l_idx = 1

    acc = gl.amd.cdna3.mfma(a, b, acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    ## --- Second half: g_idx=1, l_idx=0 ---
    g_idx = 1
    l_idx = 0

    acc = gl.amd.cdna3.mfma(a_next, b_next, acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

`a_next`/`b_next` carry data between the two halves within the same unrolled
iteration. `a`/`b` carry data into the next loop iteration.

## Step 1.4: Split the Epilogue into Two Explicit Steps

The loop now ends at `iterMax - 2`, so two epilogue steps are required:

```python
## Epilogue step 1: iteration iterMax-2 (g_idx=0, l_idx=1)
l_idx = 1
acc = gl.amd.cdna3.mfma(a, b, acc)
a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

## Epilogue step 2: iteration iterMax-1
acc = gl.amd.cdna3.mfma(a_next, b_next, acc)
```

## Step 1.5: Verify Correctness

Replace the stub imports below with your actual kernel modules:

```python
import torch

# --- REPLACE with your actual kernel imports ---
# from your_kernel_before import matmul as matmul_before
# from your_kernel_after  import matmul as matmul_after
# -----------------------------------------------

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')

c_before = matmul_before(a, b)
c_after  = matmul_after(a, b)
assert torch.allclose(c_before, c_after, atol=1e-2, rtol=1e-2), "FAILED"
print("Correctness OK, max diff:", (c_before - c_after).abs().max().item())
```

## Step 1.6: Measure Performance

```bash
rocprofv3 --stats --kernel-trace -f csv -- python3 <kernel.py> 2>&1
```

**Expected improvement**: 5–15% reduction in kernel duration from fewer SALU
ops per iteration, better ILP, and improved compiler scheduling. The effect is
more pronounced when the loop trip count is high (large K) and the kernel is
already compute-bound.

## Stage 1 Fallback

If no improvement or a regression is observed:

1. Check register pressure — the unrolled loop increases live variable count
   (`a_next`, `b_next` alongside `a`, `b`); if VGPRs spill, performance drops.
2. Check that `K` being constexpr does not cause compilation issues — matrix
   sizes must be known at JIT time.
3. The Triton compiler version in use may already eliminate the modulo
   automatically; verify with `grep -c "s_and_b32\|v_and_b32"` after both
   versions compile and confirm the count dropped.
4. If unrolling was not beneficial, restore the single-body loop and do not
   proceed to Stage 2.

---

# Stage 2: N-Slice (Split B Tile into Left/Right Halves)

## When to Apply

Apply Stage 2 after Stage 1 is in place and verified. Proceed when:

1. **Profiling shows residual pipeline bubbles**: `rocprofv3` or ATT traces show
   idle cycles between MFMA groups that are not explained by A-tile DMA latency,
   indicating the single monolithic B tile is causing serialization.
2. **B tile is large enough to split**: `BLOCK_N` must be at least 256 so that
   each half (`BLOCK_N // 2`) is still large enough to keep MFMA units occupied.

## Background

With a single `BLOCK_K × BLOCK_N` B tile loaded as one DMA group, MFMA on the
full tile is monolithic — the wavefront must wait for the entire B tile before
starting any accumulation. By splitting B into `B_left` (columns `0..BLOCK_N/2`)
and `B_right` (columns `BLOCK_N/2..BLOCK_N`):

- **Finer-grained async groups**: each sub-tile is a separate `commit_group`,
  giving 4 groups per two-iteration unroll instead of 2.
- **Interleaved MFMA**: `mfma(A, B_left, acc_left)` can execute while
  `B_right` is still in flight from the DMA engine.
- **`wait_group(2)`**: allows 2 outstanding DMA groups while consuming the
  third, enabling `B_right` LDS reads to overlap `B_left` MFMA.

## Performance Gain Explanation

Splitting B into two separate async groups (`commit_group` per half) decouples
the DMA completion events for `B_left` and `B_right`. The `wait_group(2)` call
in the loop body permits the hardware to keep two DMA groups in flight while
MFMA processes the data from the completed group. Concretely: the `mfma(A,
B_right, acc_right)` instruction can issue as soon as group 2 completes, even
while groups 3 and 4 (the next iteration's A and B tiles) are still transferring.
This fills the pipeline bubbles that remain after the loop-unroll stage.

## Step 2.1: Split Shared Memory Allocation for B

**Before:**
```python
nBuffers: gl.constexpr = 2
smemA = gl.allocate_shared_memory(...)   # [nBuffers, BLOCK_M, BLOCK_K]
smemB = gl.allocate_shared_memory(...)   # [nBuffers, BLOCK_K, BLOCK_N]
```

**After:**
```python
nBuffers: gl.constexpr = 2
smemA      = gl.allocate_shared_memory(...)   # [nBuffers, BLOCK_M, BLOCK_K] — unchanged
smemB_left  = gl.allocate_shared_memory(..., [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
smemB_right = gl.allocate_shared_memory(..., [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
```

## Step 2.2: Split Accumulators

**Before:**
```python
acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)
```

**After:**
```python
acc_left  = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
acc_right = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
```

## Step 2.3: Compute Separate B Offsets

```python
b_left_offsets  = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2   # offset by N/2 columns
```

## Step 2.4: Update the B Global Load Layout

The B global load layout now describes a `BLOCK_K × (BLOCK_N/2)` tile:

```python
gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],   # 5 reg_bases for half-width
    lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
    warp_bases=[[0, 1], [0, 2]],
    block_bases=[],
    shape=[BLOCK_K, BLOCK_N // 2],
)
```

And the shared layout:
```python
sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
    [[512, 16]],
    [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 16], [0, 32], [0, 64],
        [0, 1], [0, 2], [0, 4], [0, 8],   # no [0, 128] — tile is only N/2 wide
    ],
    [],
    [BLOCK_K, BLOCK_N // 2],
)
```

## Step 2.5: Rewrite the Prologue

Issue **4 async copy groups** — A+B_left as one group and B_right as a separate
group, for each of the two double-buffer slots:

```python
g_idx = 0
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
gl.amd.cdna4.async_copy.commit_group()   # group 1: A + B_left for iter 0

gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
gl.amd.cdna4.async_copy.commit_group()   # group 2: B_right for iter 0

a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk

g_idx = 1
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
gl.amd.cdna4.async_copy.commit_group()   # group 3: A + B_left for iter 1

gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
gl.amd.cdna4.async_copy.commit_group()   # group 4: B_right for iter 1

a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk

gl.amd.cdna4.async_copy.wait_group(3)   # wait for group 1; keep groups 2, 3, 4 in flight
l_idx  = 0
a      = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)
```

## Step 2.6: Rewrite the Loop Body

Each unrolled iteration has 4 regions, alternating between the left and right
sub-tiles:

```python
for k in range(0, iterMax - 2, 2):
    ######## Region 0 (g_idx=0, l_idx=1) ########
    g_idx = 0; l_idx = 1

    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)

    gl.amd.cdna4.async_copy.wait_group(2)   # keep 2 groups in flight
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ######## Region 1 ########
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

    gl.amd.cdna4.async_copy.wait_group(2)
    a      = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    ######## Region 2 (g_idx=1, l_idx=0) ########
    g_idx = 1; l_idx = 0

    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)

    gl.amd.cdna4.async_copy.wait_group(2)
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ######## Region 3 ########
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

    gl.amd.cdna4.async_copy.wait_group(2)
    a      = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

## Step 2.7: Update the Epilogue and Store

The epilogue now stores `c_left` and `c_right` separately:

```python
gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
c_left_offsets  = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_right_offsets = c_left_offsets + BLOCK_N * stride_cn // 2   # offset N/2 in output

# ... epilogue MFMA steps for acc_left and acc_right ...

c_left  = acc_left.to(a_ptr.dtype.element_ty)
c_left  = gl.convert_layout(c_left, layout=gStoreLayoutC)
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_left_offsets, stored_value=c_left)

c_right = acc_right.to(a_ptr.dtype.element_ty)
c_right = gl.convert_layout(c_right, layout=gStoreLayoutC)
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_right_offsets, stored_value=c_right)
```

## Step 2.8: Verify Correctness

Replace the stub imports below with your actual kernel modules:

```python
import torch

# --- REPLACE with your actual kernel imports ---
# from your_kernel_stage1 import matmul as matmul_stage1
# from your_kernel_stage2 import matmul as matmul_stage2
# -----------------------------------------------

M, N, K = 4096, 4096, 4096
a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')

c1 = matmul_stage1(a, b)
c2 = matmul_stage2(a, b)
assert torch.allclose(c1, c2, atol=1e-2, rtol=1e-2), "FAILED"
print("Correctness OK, max diff:", (c1 - c2).abs().max().item())
```

## Step 2.9: Measure Performance

```bash
rocprofv3 --stats --kernel-trace -f csv -- python3 <kernel.py> 2>&1
```

**Expected improvement**: better utilization of the async copy pipeline through
finer-grained DMA groups. The `B_right` LDS read overlaps `B_left` MFMA,
filling residual pipeline bubbles that persist after Stage 1.

## Stage 2 Fallback

If no improvement or a regression is observed:

1. Check register pressure — two accumulators (`acc_left`, `acc_right`) plus
   `b_left`/`b_right` increases VGPR usage; spilling negates the gain.
2. Check that `BLOCK_N // 2` remains large enough to keep MFMA units occupied;
   if the half-tile is too narrow, MFMA throughput drops.
3. If Stage 2 was not beneficial, revert to the Stage 1 kernel.
