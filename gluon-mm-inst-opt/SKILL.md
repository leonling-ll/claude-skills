---
name: gluon-mm-inst-opt
description: >
  Optimize memory access instructions in a Gluon GEMM kernel through two
  progressive steps: (1) replace gl.load/gl.store (flat pointer) with
  gl.amd.cdna3.buffer_load/buffer_store to eliminate masked-load branches and
  reduce address overhead; (2) on CDNA4 GPUs (gfx950/MI350), introduce LDS
  shared memory and replace buffer_load with gl.amd.cdna4.async_copy.buffer_load_to_shared
  (direct DMA from global to LDS), establishing the async pipeline structure
  needed for prefetch in later optimizations. Use when a Gluon kernel still
  uses gl.load/gl.store with mask= arguments.
  Usage: /gluon-mm-inst-opt
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Gluon GEMM: Memory Instruction Optimization

Optimize global memory access instructions in a Gluon GEMM kernel through two
progressive steps that match tutorial versions v1 and v2.

**Step A** (applies to all CDNA3/4 GPUs): Replace flat `gl.load`/`gl.store` with
AMD-native `buffer_load`/`buffer_store`.

**Step B** (CDNA4 / gfx950 only): Replace `buffer_load` with
`async_copy.buffer_load_to_shared` through LDS, establishing the DMA pipeline
that enables software-pipelined prefetch in later optimizations.

Reference tutorial:
- Step A → `v1_buffer_load`: `/home/leling/gfx9-gluon-tutorials/kernels/gemm/a16w16/v1_buffer_load/`
- Step B → `v2_async_copy`: `/home/leling/gfx9-gluon-tutorials/kernels/gemm/a16w16/v2_async_copy/`

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

On gfx942: apply Step A only, then proceed to `/gemm-v3-lds-layout` for LDS layout
improvements that do not require async_copy.

**MFMA instruction set note:**
- gfx942 (MI300X/MI308X): `version=2, instr_shape=[16, 16, 16]` in `AMDMFMALayout`
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
| `async_copy` present | Both steps done — proceed to `/gemm-v3-lds-layout` |

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

### A.6 Measure performance

```bash
touch /tmp/t0
rocprofv3 --stats --kernel-trace -f csv -- python3 <kernel.py> 2>&1
# Query avg_us from results.db (see /kernel-trace-analysis for SQL)
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

### B.4 Measure performance

```bash
touch /tmp/t0
rocprofv3 --stats --kernel-trace -f csv -- python3 <kernel.py> 2>&1
```

**Expected behavior**: Step B alone may show equal or slightly worse performance
versus Step A because `wait_group(0)` is still synchronous. This is normal — the
structural value of Step B is unlocked in `/gemm-v4-global-prefetch` (double buffering
+ `wait_group(1)`).

### B.5 Check LDS bank conflicts

The trivial `SwizzledSharedLayout(1, 1, 1, ...)` will likely cause LDS bank conflicts.
Proceed to `/gemm-v3-lds-layout` to fix them before adding prefetch.

```
/lds-bank-conflict python3 <kernel.py>
```

---

## Summary of Changes

| | Step A (buffer_load) | Step B (async_copy) |
|-|---------------------|---------------------|
| Load API | `gl.amd.cdna3.buffer_load` | `gl.amd.cdna4.async_copy.buffer_load_to_shared` |
| Data destination | Registers | LDS |
| Synchronization | Implicit | `commit_group` + `wait_group(0)` |
| LDS read | — | `load_shared_relaxed` |
| `convert_layout` needed | Yes (explicit) | No (implicit in `load_shared_relaxed`) |
| GPU requirement | CDNA3 + CDNA4 | CDNA4 only |
| Primary benefit | Eliminate branches | Enables async pipeline |

