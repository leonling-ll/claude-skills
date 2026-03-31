---
name: kernel-perf-analysis
description: >
  Collect AMD GPU kernel performance metrics for any Python kernel file using
  rocprofv3 and rocprof-compute. Four modes auto-selected based on user intent:
  (1) general performance table — VGPRs, spill count, MFMA efficiency, average
  kernel time — triggered by "how fast is", "benchmark", "VGPR", "perf table",
  or "compare versions"; (2) hardware performance counter
  collection — SQ_LDS_BANK_CONFLICT, TCC cache counters, any PMC counter —
  triggered by "counter", "bank conflict", "cache hit", "PMC", or any hardware
  counter name; (3) ATT trace collection and analysis — MFMA efficiency,
  loop/iteration/prologue/epilogue durations, bottleneck identification —
  triggered by "trace", "ATT", "MFMA efficiency", "iteration duration",
  "bottleneck", "lgkmcnt", or "vmcnt"; (4) full memory-hierarchy analysis —
  HBM, L2, L1 bandwidth and roofline, hit rates, in-flight budget (Little's
  Law) — triggered by "memory", "bandwidth", "roofline", "HBM", "L2", "L1",
  "cache hit rate", or "memory analysis". All modes accept a plain Python kernel
  file as input and produce markdown tables. Scripts live in the skill directory.
  Usage: /kernel-perf-analysis
---

# Kernel Performance Analysis

Collect AMD GPU kernel performance data for any Python kernel file. Each
required mode runs as an **independent agent on a dedicated GPU card**, then
results are presented one by one followed by a comprehensive summary.

---

## Directory Structure

```
kernel-perf-analysis/
├── SKILL.md
├── agents/
│   ├── perf-benchmarking.md      # Agent: timing + MFMA efficiency table
│   ├── perf-counter.md           # Agent: hardware counter collection
│   ├── perf-trace-analysis.md    # Agent: ATT trace + MFMA loop analysis
│   ├── perf-memory-analysis.md   # Agent: memory hierarchy BW + roofline
│   └── att-runner.md             # Legacy low-level runner (kept for reference)
├── scripts/
│   ├── run_perf_table.py         # Mode 1: parse kernel_trace.csv → perf table
│   ├── run_counter_collection.py # Mode 2: parse counter_collection.csv → table
│   ├── run_att.py                # Mode 3: run process_json.py on ui_* dir
│   └── process_json.py           # ATT post-processor: MFMA efficiency, timing
└── references/
    ├── lds-analysis-and-optimization.md  # LDS throughput & layout reference
    └── memory-bandwidth-model.md         # HBM/L2/L1 bandwidth model & roofline
```

---

## Step 0 — Determine which modes to run

### Mode selection logic

| User mentions | Modes to run |
|---------------|-------------|
| "how fast", "benchmark", "VGPR", "perf table", "compare" | Mode 1 |
| "counter", "bank conflict", "cache", "PMC", any counter name | Mode 2 |
| "trace", "ATT", "MFMA efficiency", "bottleneck", "lgkmcnt", "vmcnt" | Mode 3 |
| "memory", "bandwidth", "roofline", "HBM", "L2", "L1", "cache hit rate", "memory analysis" | Mode 4 |
| Multiple of the above | All matching modes |
| No specific hint | Mode 1 (default) |

Collect all required modes before proceeding. Multiple modes → run in parallel.

---

## Step 1 — Detect available GPUs

Before spawning agents, check how many GPU devices are available:

```bash
# Count GPUs visible to ROCm
GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo 1)
```

Assign GPU devices round-robin, starting from device 0:
- Mode 1 → GPU `0`
- Mode 2 → GPU `1` (or `0` if only one GPU)
- Mode 3 → GPU `2` (or `0` if only one GPU, or `1` if two GPUs)
- Mode 4 → GPU `3` (wraps around: `3 % GPU_COUNT`)

Formula: `GPU_FOR_MODE_N = (N-1) % GPU_COUNT`

Each agent receives its assigned `CUDA_VISIBLE_DEVICES` value and sets it
before every rocprofv3 invocation, ensuring workloads on different GPUs do not
interfere with each other's timing.

---

## Step 2 — Spawn mode agents in parallel

**CRITICAL:** Spawn all required mode agents in a **single Agent tool
message** (multiple tool calls in one response) so they run in parallel.

For each required mode, spawn a `general-purpose` agent with the instructions
from the corresponding agent file and this task description:

```
Kernel file: <absolute path to kernel.py>
Kernel name: <regex matching kernel function name>
Label: <short label, e.g. filename stem>
Output directory: /tmp/kperf_m<N>_<label>_<timestamp>
GPU device: <assigned device index>
[Mode 2 only] Counters: <comma-separated counter names>
[Mode 3 only] Iteration: [15]
[Mode 4 only] Workload name: <label with underscores, no spaces>
Iters: 20
```

Determine the kernel name regex by inspecting the kernel file:
```bash
grep -n "def \|@triton\|@gl\." <kernel_file> | head -20
```
Pick the function name decorated with `@triton.jit` or `@gl.kernel`.

### Agent file to use per mode

| Mode | Agent file |
|------|-----------|
| Mode 1 | `agents/perf-benchmarking.md` |
| Mode 2 | `agents/perf-counter.md` |
| Mode 3 | `agents/perf-trace-analysis.md` |
| Mode 4 | `agents/perf-memory-analysis.md` |

Pass the **full contents** of the agent file as the agent's system prompt /
instructions.

---

## Step 3 — Wait for all agents to finish

Wait for every spawned agent to return its `MODE<N>_RESULT` block. Do not
begin presenting results until all agents have responded.

---

## Step 4 — Present results one by one

After all agents finish, print results in mode order. Separate each section
with a horizontal rule.

### Mode 1 result block

Print the table verbatim from the agent's `TABLE:` field:

```
## Mode 1 — Benchmarking

| Version | VGPRs | Spills | MFMA Eff. | avg time |
|---------|-------|--------|-----------|----------|
| <label> | ...   | ...    | ...       | ...      |
```

If STATUS=failed, print:
```
## Mode 1 — Benchmarking
⚠ Collection failed: <ERRORS content>
```

---

### Mode 2 result block

```
## Mode 2 — Counter Collection

| Version | SQ_LDS_BANK_CONFLICT | SQ_LDS_DATA_FIFO_FULL | ... | Dispatches |
|---------|---------------------|----------------------|-----|------------|
| <label> | ...                 | ...                  | ... | ...        |
```

---

### Mode 3 result block

```
## Mode 3 — Trace Analysis

<SUMMARY content from agent>
```

---

### Mode 4 result block

Print the BW table, the roofline table, then the in-flight note:

```
## Mode 4 — Memory Hierarchy Analysis

**GPU:** <GPU_ARCH>  |  **Kernel duration:** <DURATION_NS> ns

### Memory Bandwidth

| Level | Achieved BW (GB/s) | Peak BW (GB/s) | Utilization | Hit Rate |
|-------|--------------------|----------------|-------------|----------|
| HBM   | ...                | ...            | ...%        | —        |
| L2    | ...                | ...            | ...%        | ...%     |
| L1    | ...                | ...            | ...%        | ...%     |
| LDS   | (see conflicts)    | —              | —           | —        |

**LDS bank conflicts:** <LDS_BANK_CONFLICTS>

### Roofline Analysis

| Metric               | Value                  |
|----------------------|------------------------|
| Arithmetic Intensity | <AI> FLOPs/byte        |
| Ridge Point          | <ridge> FLOPs/byte     |
| Peak Compute         | <peak_compute> TFLOPS  |
| Peak HBM BW          | <peak_hbm> GB/s        |
| **Kernel is**        | **<compute-bound / memory-bound>** |
| Bottleneck           | <BOUND_BY>             |

**In-flight budget:** <INFLIGHT_BUDGET_UTIL_PCT>% of TCP capacity in use
> <INFLIGHT_NOTE>
```

If STATUS=failed, print:
```
## Mode 4 — Memory Hierarchy Analysis
⚠ Collection failed: <ERRORS content>
```

---

## Step 5 — Comprehensive summary and suggestions

After presenting all individual results, produce a `## Summary and
Optimization Suggestions` section. Ground every suggestion in the relevant
reference documents:
- `references/lds-analysis-and-optimization.md` for LDS analysis
- `references/memory-bandwidth-model.md` for memory hierarchy and roofline analysis

Use the following decision tree:

### 5.1 MFMA Efficiency (from Mode 1 or Mode 3)

| MFMA Eff. | Assessment | Suggested skill |
|-----------|-----------|-----------------|
| > 80% | Compute-bound; well-optimized | Consider `/gluon-gpr-opt` or `/gluon-beyond-loop-opt` |
| 60–80% | Partially memory-bound | `/gluon-pipeline-opt` to improve prefetch depth |
| < 60% | Memory-bound | `/gluon-pipeline-opt` first; then `/gluon-lds-opt` |

### 5.2 LDS Bank Conflicts (from Mode 2)

Use the reference thresholds:
- `SQ_LDS_BANK_CONFLICT` > 0 per dispatch → bank conflicts present
- Steady-state `ds_read_b128` > 16 cycles in ATT → confirms conflict severity:
  - 32 cycles → 2-way conflict
  - 64 cycles → 4-way conflict

**Recommended fix priority:**
1. On gfx950 (MI350, 160 KB LDS): switch to `PaddedSharedLayout` first
   (preserves single base VGPR, negligible LDS overhead).
2. On gfx942 (MI300X, 64 KB LDS): evaluate `SwizzledSharedLayout` if padding
   would exceed LDS budget; accept extra base VGPRs as the trade-off.
3. Do **not** add more `ds_read` prefetch iterations to hide bank-conflict
   latency — this creates back-pressure without fixing throughput.

Apply with: `/gluon-lds-opt`

### 5.3 Kernel timing (from Mode 1)

- High avg time + low MFMA efficiency → bottleneck is memory latency/bandwidth
- High avg time + high MFMA efficiency → arithmetic is the bottleneck;
  consider tile size tuning or `/gluon-gpr-opt`

### 5.4 Memory hierarchy and Roofline (from Mode 4)

#### 5.4.1 Roofline verdict (primary bound classification)

The roofline verdict from Mode 4 is the **primary signal** for determining
whether the kernel is compute-bound or memory-bound:

| `ROOFLINE_VERDICT` | Interpretation | Priority action |
|--------------------|---------------|-----------------|
| `compute-bound` | AI ≥ ridge point; limited by compute throughput | Improve MFMA utilization (`/gluon-gpr-opt`, `/gluon-pipeline-opt`); if MFMA eff already high, kernel is well-optimized |
| `memory-bound` | AI < ridge point; limited by memory bandwidth/latency | Improve in-flight depth (`/gluon-pipeline-opt`); maximize L2 reuse (`/gluon-beyond-loop-opt`) |
| `N/A` | FLOP counters unavailable; fall back to MFMA efficiency from Mode 1/3 | Use MFMA efficiency decision tree (Section 5.1) |

Note: if `FLOP_SOURCE=SQ_INSTS_VALU (fallback)`, treat the arithmetic intensity
as an **upper bound** (VALU includes non-MFMA ops). The verdict is still useful
directionally but add a caveat in the summary.

#### 5.4.2 Secondary: bandwidth utilization thresholds

Use `references/memory-bandwidth-model.md` thresholds to identify the specific
memory bottleneck within a memory-bound kernel:

| Observation | Assessment | Recommended action |
|-------------|-----------|-------------------|
| HBM util > 85% | HBM-saturated | Maximize pipeline depth; check L2 hit rate |
| HBM util 50–85% | Good HBM utilization | Minor tuning; look at L2/L1 for secondary wins |
| HBM util < 50% | Request issue rate too low | Increase `num_stages`; check occupancy vs. SIMD sharing |
| L2 hit rate > 80% | Good L2 reuse | Focus on HBM or compute |
| L2 hit rate < 40% | Every access reaches HBM | Improve B-matrix reuse: `/gluon-beyond-loop-opt` (XCD remap) |
| L1 hit rate < 50% | Working set > 32 KB per CU | Reduce tile size to fit TCP; check access stride |
| In-flight budget < 50% | Pipeline under-filled | Increase `num_stages` or adjust `waves_per_simd` |
| In-flight budget ≈ 100% | TCP-limited (32 KB cap hit) | Reduce `data_per_request_per_wave` (smaller tile) |

### 5.5 Loop structure (from Mode 3)

- `pro_ratio` > 10% → prologue is too long; check if async DMA warmup is
  amortized correctly
- `epi_ratio` > 5% → epilogue tail latency; consider `/gluon-beyond-loop-opt`
- `loop_ratio` < 85% → significant time outside the K-loop

### 5.6 Summary format

```
## Summary and Optimization Suggestions

**Overall assessment:** <one sentence — lead with the roofline verdict if Mode 4 was run>

| Metric | Value | Status |
|--------|-------|--------|
| **Roofline verdict** | **<compute-bound / memory-bound / N/A>** | — |
| Arithmetic intensity | <AI> FLOPs/byte (ridge: <ridge>) | ✅ compute-bound / ⚠ memory-bound |
| MFMA efficiency | <value> | ✅ / ⚠ / ❌ |
| LDS bank conflicts | <count/dispatch> | ✅ / ⚠ / ❌ |
| avg kernel time | <value> | — |
| loop ratio | <value> | ✅ / ⚠ |
| HBM utilization | <value>% | ✅ / ⚠ / ❌ |
| L2 hit rate | <value>% | ✅ / ⚠ / ❌ |
| L1 hit rate | <value>% | ✅ / ⚠ / ❌ |
| In-flight budget used | <value>% of TCP | ✅ / ⚠ / ❌ |

**Recommended next steps (in priority order):**
1. <step 1 with skill name and rationale grounded in reference>
2. <step 2 ...>
...

**References:**
- `references/lds-analysis-and-optimization.md` — LDS throughput model & layout
- `references/memory-bandwidth-model.md` — HBM/L2/L1 bandwidth model & roofline
```

---

## Mode Selection Reference

| Counter | Cache level | Measures | Threshold |
|---------|------------|---------|-----------|
| `SQ_LDS_BANK_CONFLICT` | LDS | LDS bank conflicts | 0 = clean; > 0 = conflicts |
| `SQ_LDS_DATA_FIFO_FULL` | LDS | LDS data FIFO saturation | Should be 0 |
| `TCC_EA_RDREQ_DRAM_sum` | L2→HBM | L2→DRAM read requests (64B/req) | High = HBM-bound |
| `TCC_EA_WRREQ_DRAM_sum` | L2→HBM | L2→DRAM write requests (64B/req) | — |
| `TCC_HIT_sum` | L2 | L2 read hits | High = good L2 reuse |
| `TCC_MISS_sum` | L2 | L2 read misses → fetched from HBM | High = HBM pressure |
| `TCP_TCC_READ_REQ_sum` | L1→L2 | L1 misses forwarded to L2 | High = L1 pressure |
| `TCP_TOTAL_CACHE_ACCESSES_sum` | L1 | Total L1 access requests from shader | Base for L1 hit rate |
| `GRBM_GUI_ACTIVE` | GPU | GPU active cycles | — |

### rocprof-compute GUI (Mode 4 only)

After Mode 4 collection, the user can launch an interactive GUI to explore
the full memory hierarchy breakdown:

```bash
rocprof-compute analyze \
  -p <output_dir>/<workload_name>/<GPU_MODEL>/ \
  --gui -R BF16
```

Run this command in a VSCode terminal, then forward the port to your local
browser. The `-R BF16` flag selects the BF16 roofline filter.
