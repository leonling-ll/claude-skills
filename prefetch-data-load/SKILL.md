---
name: prefetch-data-load
description: >
  Apply prefetch optimization to Triton/Gluon kernel loops: pre-load the first
  iteration's data before the loop, issue async loads for the next iteration
  inside the loop body, and swap buffers at the loop tail. This overlaps data
  load latency with compute instructions. Use when a kernel has a loop where
  gl.load/tl.load feeds into MFMA/dot/compute and load latency is exposed.
  Usage: /prefetch-data-load
tools: Read,Edit,Bash,Grep,Glob,Agent
---

# Prefetch Data Load Optimization

Apply software prefetch (double-buffering) to overlap async data loads with
compute in Triton/Gluon GPU kernel loops.

## Core Principle

GPU global memory loads (`gl.load`, `tl.load`, `gl.amd.cdna3.buffer_load`)
are **asynchronous** -- the load instruction returns immediately and the
hardware fetches data in the background. The data is only needed when a
subsequent instruction actually **consumes** it. If we issue the load early
enough, the data arrives by the time we need it, effectively hiding the load
latency behind compute work.

**Without prefetch** (load latency fully exposed):

```
for i in range(N):
    data = load(ptr + i)     # <-- stall: wait for data
    result = compute(data)   # <-- cannot start until load completes
```

Timeline:
```
|--load--|--stall--|--compute--|--load--|--stall--|--compute--|
```

**With prefetch** (load overlapped with compute):

```
# Pre-load first iteration BEFORE the loop
next_data = load(ptr + 0)

for i in range(N):
    # Swap: the prefetched data becomes current
    data = next_data

    # Issue load for NEXT iteration (async, non-blocking)
    if i + 1 < N:
        next_data = load(ptr + i + 1)

    # Compute using CURRENT data -- overlaps with next load
    result = compute(data)
```

Timeline:
```
|--load₀--|--compute₀ + load₁--|--compute₁ + load₂--|--compute₂--|
```

The total time drops from `N * (load + compute)` to roughly
`load + N * max(load, compute)`.

## Transformation Steps

Given a loop like:

```python
for i in range(START, END):
    # === LOAD PHASE ===
    offsets = compute_offsets(i)
    data_A = gl.load(ptr_A + offsets)
    data_B = gl.load(ptr_B + offsets)

    # === COMPUTE PHASE ===
    result = gl.amd.cdna3.mfma(transform(data_A), transform(data_B), acc)
```

Apply the following mechanical transformation:

### Step 1: Hoist first load before loop

```python
# --- Pre-load iteration START ---
offsets_0 = compute_offsets(START)
next_data_A = gl.load(ptr_A + offsets_0)
next_data_B = gl.load(ptr_B + offsets_0)

for i in range(START, END):
    ...
```

### Step 2: Swap at loop top

```python
for i in range(START, END):
    # Swap: prefetched -> current
    data_A = next_data_A
    data_B = next_data_B
    ...
```

### Step 3: Issue next iteration's load BEFORE compute

```python
for i in range(START, END):
    data_A = next_data_A
    data_B = next_data_B

    # Prefetch next iteration (guarded)
    if i + 1 < END:
        offsets_next = compute_offsets(i + 1)
        next_data_A = gl.load(ptr_A + offsets_next)
        next_data_B = gl.load(ptr_B + offsets_next)

    # Compute using current data (overlaps with next load)
    result = gl.amd.cdna3.mfma(transform(data_A), transform(data_B), acc)
```

### Step 4: Handle offset/auxiliary computation for next iteration

Any offset calculations, block table lookups, or scale factor loads needed
for the *next* iteration's data should also be issued early (before compute),
since they may themselves involve memory loads:

```python
for i in range(START, END):
    data_A = next_data_A
    data_B = next_data_B
    block_id = next_block_id
    scale = next_scale

    if i + 1 < END:
        next_block_id = load_block_table(i + 1)
        offsets_next = compute_offsets(i + 1, next_block_id)
        next_data_A = gl.load(ptr_A + offsets_next)
        next_data_B = gl.load(ptr_B + offsets_next)
        next_scale = gl.load(scale_ptr + next_block_id)

    result = gl.amd.cdna3.mfma(transform(data_A) * scale, transform(data_B), acc)
```

## Concrete Example: Paged Attention KV Loop

### Before (load latency exposed)

```python
for kv_idx in range(KV_BLOCK_COUNT):
    kv_start = partition_start + kv_idx * BLOCK_SIZE

    # Block table lookup
    block_table_id = kv_start // KV_BLOCK_SIZE
    kv_page_id = tl.load(block_table_ptr + block_table_id).to(gl.int64)

    # Load key
    key_offsets = kv_page_id * stride_block + kv_head_idx * stride_head + ...
    key_block = gl.load(key_cache_ptr + key_offsets)  # <-- STALL until data arrives

    # Load value
    val_offsets = kv_page_id * stride_block + ...
    val_block = gl.load(value_cache_ptr + val_offsets)  # <-- STALL

    # Load KV scales (FP8)
    if KV_QUANT_MODE == 1:
        k_scale = gl.load(key_scale + scale_offsets)
        v_scale = gl.load(value_scale + scale_offsets)

    # Reshape key for MFMA
    key_block = gl.permute(key_block, [1, 3, 0, 2])
    key_block = gl.reshape(key_block, [HEAD_SIZE, BLOCK_SIZE])

    # QK MFMA -- compute cannot overlap with loads above
    qk_acc = gl.zeros(...)
    qk = gl.amd.cdna3.mfma(query, key_block, qk_acc)

    # Reshape value
    val_block = gl.permute(val_block, [0, 2, 1])
    val_block = gl.reshape(val_block, [BLOCK_SIZE, HEAD_SIZE])

    # Softmax + PV MFMA
    probs = softmax(qk)
    output = gl.amd.cdna3.mfma(probs, val_block, pv_acc)
```

### After (prefetch applied)

```python
# ====== PRE-LOAD: first iteration ======
kv_start_0 = partition_start
block_table_id_0 = kv_start_0 // KV_BLOCK_SIZE
next_kv_page_id = tl.load(block_table_ptr + block_table_id_0).to(gl.int64)

next_key_offsets = next_kv_page_id * stride_block + kv_head_idx * stride_head + ...
next_key_block = gl.load(key_cache_ptr + next_key_offsets)

next_val_offsets = next_kv_page_id * stride_block + ...
next_val_block = gl.load(value_cache_ptr + next_val_offsets)

if KV_QUANT_MODE == 1:
    next_k_scale = gl.load(key_scale + scale_offsets_0)
    next_v_scale = gl.load(value_scale + scale_offsets_0)

# ====== MAIN LOOP ======
for kv_idx in range(KV_BLOCK_COUNT):
    # ------ Swap: prefetched -> current ------
    kv_page_id = next_kv_page_id
    key_block = next_key_block
    val_block = next_val_block
    if KV_QUANT_MODE == 1:
        k_scale = next_k_scale
        v_scale = next_v_scale

    # ------ Prefetch NEXT iteration (async) ------
    if kv_idx + 1 < KV_BLOCK_COUNT:
        kv_start_next = partition_start + (kv_idx + 1) * BLOCK_SIZE
        block_table_id_next = kv_start_next // KV_BLOCK_SIZE
        next_kv_page_id = tl.load(block_table_ptr + block_table_id_next).to(gl.int64)

        next_key_offsets = next_kv_page_id * stride_block + kv_head_idx * stride_head + ...
        next_key_block = gl.load(key_cache_ptr + next_key_offsets)

        next_val_offsets = next_kv_page_id * stride_block + ...
        next_val_block = gl.load(value_cache_ptr + next_val_offsets)

        if KV_QUANT_MODE == 1:
            next_k_scale = gl.load(key_scale + scale_offsets_next)
            next_v_scale = gl.load(value_scale + scale_offsets_next)

    # ------ Compute (overlaps with prefetch loads) ------
    # Reshape key
    key_block = gl.permute(key_block, [1, 3, 0, 2])
    key_block = gl.reshape(key_block, [HEAD_SIZE, BLOCK_SIZE])

    # QK MFMA -- next key/value load runs concurrently
    qk_acc = gl.zeros(...)
    qk = gl.amd.cdna3.mfma(query, key_block, qk_acc)

    # Reshape value
    val_block = gl.permute(val_block, [0, 2, 1])
    val_block = gl.reshape(val_block, [BLOCK_SIZE, HEAD_SIZE])

    # Softmax + PV MFMA
    probs = softmax(qk)
    output = gl.amd.cdna3.mfma(probs, val_block, pv_acc)
```

## Applicable Patterns

This optimization applies whenever you see this pattern in a kernel:

| Signal | Description |
|--------|-------------|
| `for ... in range(N)` loop with `gl.load`/`tl.load` followed by MFMA/dot | Load-then-compute in a loop body |
| Block table lookup inside loop | `tl.load(block_table + idx)` followed by `gl.load(cache + page_id * stride)` |
| KV cache iteration | Paged attention, flash attention, any tiled GEMM with paged memory |
| Scale factor loads | FP8 per-token quantization scales loaded per KV block |

## Gluon Kernel Constraints

When applying prefetch to Gluon kernels (Triton's AMD GPU backend), be aware:

### Register Budget
Gluon kernels compile to GCN ISA where `s_waitcnt` insertion is controlled by the **compiler**, not by the programmer. You cannot directly eliminate `s_waitcnt` instructions. Instead, prefetch restructures the code so the compiler places `s_waitcnt` after enough compute work to hide the latency.

**Always check register headroom before adding prefetch buffers:**

On CDNA3 (gfx942 MI300X/MI308), VGPRs are split into **two separate register files**:
- **arch_vgpr** (256 per SIMD): used by VALU, VMEM loads, LDS ops, and prefetch buffers
- **accum_vgpr / AGPR** (256 per SIMD): used exclusively by MFMA result writeback

Prefetch buffers consume **arch_vgpr only** (they hold global load results). MFMA accumulators use **accum_vgpr only**. These do not compete.

```python
# Estimate arch_vgpr cost of prefetch buffers:
#   - Each global_load_dwordx4 = 4 arch_vgpr per load
#   - 8 K-cache loads = 8 × 4 = 32 arch_vgpr for one buffer set
#   - Double-buffering = 2 × 32 = 64 arch_vgpr (but one set is reused)
#   - Net additional arch_vgpr ≈ 32 (the "next" buffer)
#
# On MI300X (gfx942): 256 arch_vgpr + 256 accum_vgpr per SIMD
# Occupancy = 256 / max(arch_vgpr, accum_vgpr) waves per SIMD
#
# Example: arch=148, accum=148 → occupancy bottleneck = 148 → 1 wave
# Adding 32 arch_vgpr → arch=180, accum=148 → bottleneck = 180 → still 1 wave (safe)
# Adding 120 arch_vgpr → arch=268 → SPILL (critical, exceeds 256 per SIMD)
```

**Critical thresholds (gfx942, per register file):**
| Register File | Count | Max Waves/SIMD | Impact |
|--------------|-------|---------------|--------|
| arch_vgpr ≤ 128 | or accum ≤ 128 | 2 | Good occupancy |
| arch_vgpr ≤ 256 | or accum ≤ 256 | 1 | Minimum occupancy |
| arch_vgpr > 256 | or accum > 256 | **SPILL** | Register overflow → severe perf regression |

**How to check current VGPR allocation** (from rocprofv3 database):
```sql
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
WHERE ks.KernelName LIKE '%target_kernel%'
LIMIT 5;
```

**WARNING**: Do NOT use `maxnreg` to force `accum_vgpr=0` in hopes of freeing register space for prefetch. This forces MFMA results through arch_vgpr via `v_accvgpr_read` spills, causing massive slowdown (measured 4.5x GPU kernel regression). See `/optimize-pa-decode-gluon` Section 1.2.1.

### What Prefetch Can and Cannot Do in Gluon

**CAN do:**
- Restructure the Python-level loop so `gl.load` / `tl.load` is issued earlier
- The compiler will then schedule the corresponding `s_waitcnt` further from the load
- Overlap next iteration's loads with current iteration's MFMA compute

**CANNOT do:**
- Directly control `s_waitcnt vmcnt(N)` counter values
- Force the compiler to use `vmcnt(N>0)` instead of `vmcnt(0)`
- Eliminate barriers (`s_barrier`) — these come from explicit `tl.debug_barrier()` or Gluon's reduce primitives

### Hoisting Loads into Barrier-Wait Regions

A powerful technique specific to multi-phase kernels (like paged attention with softmax reduce):

If a kernel has a phase that spends time in `s_barrier` waits (e.g., softmax cross-wave reduce), and the **next** phase needs data from global memory (e.g., V-value loads), hoist those loads into the barrier-stalling region. The barrier must wait regardless — issuing loads during that wait is essentially free.

```python
# BEFORE: V-value loads happen AFTER softmax reduce completes
softmax_reduce(qk_scores)  # <-- 96K stall cycles in barriers
v_data = gl.load(v_ptr + offsets)  # <-- additional load latency

# AFTER: V-value loads issued BEFORE/DURING softmax reduce
v_data_prefetch = gl.load(v_ptr + offsets)  # <-- async, non-blocking
softmax_reduce(qk_scores)  # <-- barrier stalls now overlap with v_data fetch
v_data = v_data_prefetch  # <-- data likely already arrived
```

This works because:
- `gl.load` returns immediately (async)
- The barrier stalls are **dead time** where no useful work happens
- By the time barriers complete (~96K cycles), the V-value load (~17K cycles) has long since arrived

## Rules and Pitfalls

### Do
- **Prefetch ALL data** needed for the next iteration: keys, values, scales, block table entries
- **Place prefetch loads** immediately after the swap, BEFORE any compute that consumes current data
- **Guard prefetch** with `if i + 1 < END` to avoid out-of-bounds loads on the last iteration
- **Minimize work between load and consume**: the more compute between prefetch issue and data use, the better the overlap
- **Keep the swap simple**: just variable assignment, no computation
- **Check VGPR budget**: on MI308, calculate `current_vgpr + current_agpr + prefetch_vgprs ≤ 512` to avoid spills
- **Hoist cross-phase loads into barrier regions**: if a kernel has barrier-heavy phases (reduce/sync), issue the next phase's loads before/during those barriers

### Don't
- **Don't prefetch if loop body is already memory-bound**: prefetching helps when compute (MFMA) duration >= load latency. If the loop is purely loads with no compute, prefetching won't help.
- **Don't prefetch too many buffers**: each prefetched variable occupies registers. If register pressure is already high (causing spills), prefetching more data makes it worse. Check `waves_per_eu` / occupancy.
- **Don't assume occupancy can increase**: on MI308 with 512 max VGPRs, adding prefetch buffers that push total VGPRs above 256 will drop occupancy from 2 to 1 wave/SIMD. This may or may not be acceptable — profile both configurations.
- **Don't reorder loads that have data dependencies**: if `load_B` depends on the result of `load_A` (e.g., block table lookup -> cache load), they must stay sequential within the prefetch block.
- **Don't forget to handle `KV_QUANT_MODE` branches**: if scale loads are conditional, the prefetch must replicate the same conditions.
- **Don't break the first-iteration / last-iteration semantics**: the pre-loop load covers iteration 0; the in-loop prefetch covers iteration `i+1`. After the loop ends, `next_*` variables hold stale data from the guard-failed last iteration -- this is fine since they're never consumed.

## Verification

After applying prefetch:

1. **Correctness**: Run the existing test suite. Output must match bit-for-bit (fp32 accumulation) or within tolerance (fp8/bf16).
2. **Performance**: Profile with `rocprofv3` or Triton's built-in profiler. Look for:
   - Reduced `VMEM` stall cycles in the loop body
   - Higher MFMA utilization percentage
   - Overall kernel duration reduction
3. **Register pressure**: Check that `waves_per_eu` (occupancy) didn't drop. If it did, consider prefetching fewer buffers (e.g., only keys, not values).

## FlyDSL: scf.for with Loop-Carried Prefetch

In FlyDSL kernels, Python-level `for _pi in range(N)` gets traced into N flat copies that LLVM re-rolls. This makes the `data = next_data` swap **invisible** to MLIR — both variables alias the same SSA value, so LLVM hoists loads as loop-invariant.

**Solution**: Use FlyDSL's `scf.for` with `init=` (loop-carried values) to create genuine SSA phi nodes. See the `flydsl-kernel-authoring` skill, section "scf.for with Loop-Carried Values", for the full pattern and three critical pitfalls.

### PA Decode Kernel Example (verified, 112us → 0.75x Gluon)

State inventory (15 values carried across iterations):
- 8 × `vector<4xi32>` — K data (4 tiles × 2 loads)
- 1 × `i32` — partition_start
- 2 × `i32` — block table values (phys_block/page_off or phys_0/phys_1)
- 2 × `f32` — running_max, running_sum (online softmax)
- 2 × `vector<4xf32>` — PV accumulators

```python
# Pack/unpack helpers
def _pack(kv_flat, part_start, bt_vals, rmax, rsum, acc_pv):
    raw = kv_flat + [part_start] + bt_vals + [rmax, rsum] + acc_pv
    return [v.ir_value() if hasattr(v, 'ir_value') else v for v in raw]

def _unpack(state):
    kv_flat = list(state[0:8])
    kv = [[kv_flat[t*2], kv_flat[t*2+1]] for t in range(4)]
    return kv, state[8], list(state[9:11]), state[11], state[12], [state[13], state[14]]

# Prologue
pf_0 = issue_bt_k_loads(partition_0)
init_state = _pack(flatten(pf_0['kv']), pf_0['part_start'], ...)

# scf.for (bounds MUST be arith.index, not Python ints!)
for iv, state in range(arith.index(0), arith.index(N-1), arith.index(1), init=init_state):
    kv, part_start, bt, rmax, rsum, acc = _unpack(state)
    rmax, rsum, acc = compute_qk_softmax_pv(kv, part_start, bt, rmax, rsum, acc)
    pf_next = issue_bt_k_loads(next_partition(iv + 1))
    results = yield _pack(flatten(pf_next['kv']), pf_next['part_start'], ...)

# Epilogue: clear SmemPtr caches, compute last partition, write output
smem_ptr._view_cache = None
kv, part_start, bt, rmax, rsum, acc = _unpack(results)
compute_qk_softmax_pv(kv, part_start, bt, rmax, rsum, acc)
write_output(rmax, rsum, acc)
```

**ISA result**: 8 K-prefetch `buffer_load_dwordx4` appear at the END of the loop body (after PV MFMA), overlapping with the MFMA pipeline drain. The prologue has 8 K loads before the loop. The epilogue has 8 V loads only (no K loads needed).

### Key Differences from Triton/Gluon Prefetch

| Aspect | Triton/Gluon | FlyDSL |
|--------|-------------|--------|
| Loop type | Python `for` → LLVM re-rolled | `scf.for` with MLIR phi nodes |
| Swap mechanism | Python `data = next_data` | `yield` → `scf.yield` → block args |
| Compiler visibility | LLVM sees aliased SSA values | MLIR sees distinct block args per iteration |
| Guard for last iter | `if i + 1 < N` inside loop | Prologue/epilogue pattern (loop has N-1 iters) |
| State packing | N/A (Python variables) | Flat list of unwrapped `ir.Value`s |

## When NOT To Use

- **Single-iteration loops** (`range(1)`): no next iteration to prefetch
- **Already-pipelined kernels**: if the kernel uses `num_stages > 1` in Triton, the compiler may already handle software pipelining -- adding manual prefetch on top can conflict
- **Compute-bound kernels**: if MFMA utilization is already >90%, the bottleneck is compute, not memory -- prefetching won't help
- **Very high register pressure**: if occupancy is already 1 wave/EU and the kernel spills, adding prefetch buffers will make it worse
