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

Two progressive optimizations for a Gluon GEMM kernel that already has
double-buffered LDS and local prefetch in place:

- **Stage 1 — Loop Unroll ×2**: eliminate the `k % 2` modulo overhead by
  duplicating the loop body with hardcoded buffer indices (`g_idx=0/1`,
  `l_idx=1/0`), so the compiler can resolve all LDS slot accesses statically.
- **Stage 2 — N-Slice**: split the B tile into `B_left` and `B_right` halves
  with separate async copy groups, enabling `B_right` MFMA to overlap `B_left`
  DMA and filling residual pipeline bubbles.

Apply Stage 1 first; verify it is effective before proceeding to Stage 2.

**Both stages require CDNA4 (gfx950 / MI350)** — `gl.amd.cdna4.async_copy` must
be available. If the platform is CDNA3 (gfx942), stop here.

---

## Step 0: Check Platform and Baseline ISA

```bash
# Confirm CDNA4
python3 -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
# Expected: gfx950  (MI350). If gfx942, stop — skill does not apply.

# Find compiled kernel ISA
find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -5

# Count k%2 modulo instructions in the hot loop (Stage 1 signal)
grep -c "s_and_b32\|v_and_b32" <path>.amdgcn

# Count residual stalls (Stage 2 signal — run after Stage 1)
grep -c "s_waitcnt vmcnt(0)" <path>.amdgcn
```

Apply Stage 1 when:
- `s_and_b32` / `v_and_b32` appear **inside the main loop body** (runtime k%2 overhead), or
- ATT trace shows unexplained SALU stalls between MFMA groups not caused by LDS or DMA.

Apply Stage 2 (after Stage 1 passes verification) when:
- Residual pipeline bubbles remain between MFMA groups.
- `BLOCK_N ≥ 256` (each half-tile `BLOCK_N // 2` must be wide enough for MFMA).

---

## Background

### Stage 1: why `k % 2` costs cycles

In a double-buffered kernel the loop alternates slots via `g_idx = k % 2`. The
compiler emits this as `s_and_b32` (or `v_and_b32`) every iteration, preventing
static resolution of `smemA.index(g_idx)`. Unrolling by 2 replaces the runtime
modulo with hardcoded `g_idx ∈ {0, 1}`, allowing the compiler to resolve all
LDS slot accesses at compile time, eliminate the SALU overhead, and improve
instruction-level parallelism between MFMA and DMA.

Prerequisite: `K` must be `gl.constexpr` so that `iterMax = gl.cdiv(K, BLOCK_K)`
is a compile-time constant and the loop range `range(0, iterMax - 2, 2)` is valid.

### Stage 2: why a monolithic B tile leaves pipeline bubbles

With a single `BLOCK_K × BLOCK_N` B tile as one `commit_group`, MFMA on the full
tile cannot begin until the entire B tile lands in LDS. Splitting B into `B_left`
and `B_right` halves — each a separate `commit_group` — decouples the two DMA
completion events. `wait_group(2)` keeps two groups in flight while MFMA processes
the completed group, enabling `mfma(A, B_right, acc_right)` to issue as soon as
the B_right group completes even while later groups are still transferring.

---

## Stage 1: Loop Unroll ×2

### Code Template

#### 1. Make K a constexpr (if not already)

```python
# Before
def matmul_kernel(..., K, ...):
    iterMax = gl.cdiv(K, BLOCK_K)

# After
def matmul_kernel(..., K: gl.constexpr, ...):
    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 3)   # at least 2 full unrolled pairs before epilogue
```

The launcher passes `K` unchanged — Triton propagates constexpr automatically.

#### 2. Loop: step by 2, hardcode g_idx / l_idx per half

```python
# Before (single body, runtime indices)
for k in range(0, iterMax - 1):
    g_idx = k % 2          # runtime modulo → s_and_b32
    l_idx = 1 - g_idx

    acc = gl.amd.cdna3.mfma(a, b, acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

# After (unrolled ×2, static indices)
for k in range(0, iterMax - 2, 2):
    ## --- First half: g_idx=0, l_idx=1 ---
    acc = gl.amd.cdna3.mfma(a, b, acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(0), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
    b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    ## --- Second half: g_idx=1, l_idx=0 ---
    acc = gl.amd.cdna3.mfma(a_next, b_next, acc)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(1), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(1), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(0), dotOpLayoutB)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

`a_next` / `b_next` carry data between the two halves of one unrolled pair.
`a` / `b` carry data into the next pair.

#### 3. Epilogue: two explicit steps

The loop ends at `iterMax - 2`, leaving exactly two tiles unprocessed:

```python
## Epilogue step 1 — tile iterMax-2 (l_idx=1, data already in a, b from loop)
acc = gl.amd.cdna3.mfma(a, b, acc)
a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(1), dotOpLayoutB)

## Epilogue step 2 — tile iterMax-1
acc = gl.amd.cdna3.mfma(a_next, b_next, acc)
```

---

### Stage 1 Verification ✓

#### Correctness — compare against the pre-unroll kernel

```python
import torch, importlib.util

def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

before = load_kernel("kernel_before.py", "before")
stage1 = load_kernel("kernel_stage1.py", "stage1")

x = torch.randn(B, M, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

c_ref = before.launcher(x, w)
c_new = stage1.launcher(x, w)
assert torch.allclose(c_ref, c_new, atol=1.0, rtol=0), \
    f"Stage 1 FAILED: max diff = {(c_ref - c_new).abs().max().item()}"
print("Stage 1 correctness OK")
```

#### Performance + ISA — confirm modulo removed and kernel is faster

```python
for _ in range(10): before.launcher(x, w); stage1.launcher(x, w)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(200): before.launcher(x, w)
end.record(); torch.cuda.synchronize()
base_us = start.elapsed_time(end) / 200 * 1000

start.record()
for _ in range(200): stage1.launcher(x, w)
end.record(); torch.cuda.synchronize()
s1_us = start.elapsed_time(end) / 200 * 1000

print(f"Before:   {base_us:.1f} µs")
print(f"Stage 1:  {s1_us:.1f} µs  ({base_us/s1_us:.3f}x)")
```

```bash
s1_isa=$(find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -1 | awk '{print $NF}')
echo "VGPRs:"; grep "NumVgprs:" $s1_isa
echo "s_and_b32 count (should drop to 0 in loop body):"
grep -c "s_and_b32\|v_and_b32" $s1_isa
```

#### Decision

| Outcome | Action |
|---------|--------|
| Faster **and** `s_and_b32` count dropped | ✅ Stage 1 succeeded — proceed to Stage 2 |
| Same speed, `s_and_b32` already 0 before | ⚠️ Compiler already eliminated modulo. Proceed to Stage 2 if bubbles remain. |
| Slower, VGPRs increased significantly | ❌ Unrolled body raised live-variable count (`a_next`, `b_next`). Revert. Do not proceed. |
| Slower, VGPRs unchanged | ❌ Loop body is already compute-bound. Revert. Do not proceed. |

---

## Stage 2: N-Slice (Split B Tile into Left/Right Halves)

### Code Template

#### 1. Split shared memory and accumulators

```python
# Before
smemB = gl.allocate_shared_memory(b_ptr.type.element_ty, [nBuffers, BLOCK_K, BLOCK_N],  sharedLayoutB)
acc   = gl.zeros((BLOCK_M, BLOCK_N),     gl.float32, mfmaLayout)

# After
smemB_left  = gl.allocate_shared_memory(b_ptr.type.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
smemB_right = gl.allocate_shared_memory(b_ptr.type.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB)
acc_left    = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
acc_right   = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
```

Update `sharedLayoutB` and `dotOpLayoutB` to describe a `BLOCK_K × (BLOCK_N/2)`
tile (drop the `[0, BLOCK_N]` basis vector; halve any N-dimension extent).

#### 2. Compute left/right B offsets

```python
b_left_offsets  = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
b_right_offsets = b_left_offsets + (BLOCK_N // 2) * stride_bn
```

#### 3. Prologue: 4 commit groups, pre-read tile 0 A + B_left

```python
## Slot 0: A + B_left (group 1), B_right (group 2)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(0),      a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
gl.amd.cdna4.async_copy.commit_group()                              # group 1
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
gl.amd.cdna4.async_copy.commit_group()                              # group 2
a_base += BLOCK_K * stride_ak; b_base += BLOCK_K * stride_bk

## Slot 1: A + B_left (group 3), B_right (group 4)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(1),      a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_offsets)
gl.amd.cdna4.async_copy.commit_group()                              # group 3
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(1), b_base, b_right_offsets)
gl.amd.cdna4.async_copy.commit_group()                              # group 4
a_base += BLOCK_K * stride_ak; b_base += BLOCK_K * stride_bk

## Wait for group 1 only; keep groups 2–4 in flight
gl.amd.cdna4.async_copy.wait_group(3)
a      = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(0),      dotOpLayoutA)
b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(0), dotOpLayoutB)
```

#### 4. Loop: 4 regions per unrolled pair, interleaving left/right MFMA with DMA

```python
for k in range(0, iterMax - 2, 2):
    ## Region 0 — slot 0, consume B_left, load next A+B_left into slot 0
    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
    gl.amd.cdna4.async_copy.wait_group(2)          # keep 2 groups in-flight
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(0), dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(0),      a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ## Region 1 — consume B_right, load next B_right into slot 0, read slot 1 A+B_left
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)
    gl.amd.cdna4.async_copy.wait_group(2)
    a      = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(1),      dotOpLayoutA)
    b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(1), dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a_base += BLOCK_K * stride_ak; b_base += BLOCK_K * stride_bk

    ## Region 2 — slot 1, consume B_left, load next A+B_left into slot 1
    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
    gl.amd.cdna4.async_copy.wait_group(2)
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(1), dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(1),      a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    ## Region 3 — consume B_right, load next B_right into slot 1, read slot 0 A+B_left
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)
    gl.amd.cdna4.async_copy.wait_group(2)
    a      = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(0),      dotOpLayoutA)
    b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(0), dotOpLayoutB)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(1), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a_base += BLOCK_K * stride_ak; b_base += BLOCK_K * stride_bk
```

#### 5. Epilogue: drain remaining groups, two MFMA steps, split store

```python
## Epilogue step 1 — tile iterMax-2 (a, b_left already pre-loaded)
acc_left  = gl.amd.cdna3.mfma(a, b_left, acc_left)
gl.amd.cdna4.async_copy.wait_group(2)
b_right   = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(0), dotOpLayoutB)
a_next    = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(1),        dotOpLayoutA)
b_left_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(1), dotOpLayoutB)
acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

## Epilogue step 2 — tile iterMax-1
gl.amd.cdna4.async_copy.wait_group(0)
acc_left  = gl.amd.cdna3.mfma(a_next, b_left_next, acc_left)
b_right_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(1), dotOpLayoutB)
acc_right = gl.amd.cdna3.mfma(a_next, b_right_next, acc_right)

## Store left and right halves of C
c_left  = acc_left.to(output_ptr.type.element_ty)
c_right = acc_right.to(output_ptr.type.element_ty)
c_left_offsets  = offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
c_right_offsets = c_left_offsets + (BLOCK_N // 2) * stride_cn
gl.amd.cdna3.buffer_store(ptr=output_ptr, offsets=c_left_offsets,  stored_value=c_left,  mask=c_mask)
gl.amd.cdna3.buffer_store(ptr=output_ptr, offsets=c_right_offsets, stored_value=c_right, mask=c_mask)
```

---

### Stage 2 Verification ✓

#### Correctness — compare against the Stage 1 kernel

```python
c_s1 = stage1.launcher(x, w)
c_s2 = stage2.launcher(x, w)
assert torch.allclose(c_s1, c_s2, atol=1.0, rtol=0), \
    f"Stage 2 FAILED: max diff = {(c_s1 - c_s2).abs().max().item()}"
print("Stage 2 correctness OK")
```

#### Performance + ISA — confirm pipeline bubbles reduced

```python
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
s2_isa=$(find ~/.triton/cache -name "*.amdgcn" | xargs ls -lt | head -1 | awk '{print $NF}')
echo "VGPRs:"; grep "NumVgprs:" $s2_isa
echo "vmcnt(0) count:"; grep -c "s_waitcnt vmcnt(0)" $s2_isa
# Confirm wait_group(2) pattern is present (not wait_group(0) everywhere)
grep "wait_group" $s2_isa | sort | uniq -c
```

#### Decision

| Outcome | Action |
|---------|--------|
| Faster **and** pipeline-bubble stalls reduced | ✅ Stage 2 succeeded — keep it |
| Slower, VGPRs increased significantly | ❌ Two accumulators + `b_left`/`b_right` spilled. Revert to Stage 1. |
| Slower, VGPRs unchanged | ❌ Kernel is MFMA-bound or `BLOCK_N // 2` too narrow for the MFMA unit. Revert to Stage 1. |
| Same speed | ❌ Compiler already fills the bubbles without N-slice. Revert to Stage 1. |

---

## Performance Summary

| Stage | Eliminates | Mechanism | Expected Speedup |
|-------|-----------|-----------|-----------------|
| Stage 1 | `k%2` SALU overhead | Hardcode `g_idx ∈ {0,1}` per half; compiler resolves LDS slots statically | 5–15% |
| Stage 2 | B-tile monolithic DMA serialization | 4 `commit_group`s per unrolled pair; `wait_group(2)` keeps 2 DMA groups in-flight | 5–15% additional |
