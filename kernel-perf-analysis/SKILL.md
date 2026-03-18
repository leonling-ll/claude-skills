---
name: kernel-perf-analysis
description: >
  Collect AMD GPU kernel performance metrics for any Python kernel file using
  rocprofv3. Three modes auto-selected based on user intent: (1) general
  performance table — VGPRs, spill count, MFMA efficiency, average kernel
  time — triggered by "how fast is", "benchmark", "VGPR", "perf table", or
  "compare versions"; (2) hardware performance counter
  collection — SQ_LDS_BANK_CONFLICT, TCC cache counters, any PMC counter —
  triggered by "counter", "bank conflict", "cache hit", "PMC", or any hardware
  counter name; (3) ATT trace collection and analysis — MFMA efficiency,
  loop/iteration/prologue/epilogue durations, bottleneck identification —
  triggered by "trace", "ATT", "MFMA efficiency", "iteration duration",
  "bottleneck", "lgkmcnt", or "vmcnt". All three modes accept a plain Python
  kernel file as input and produce markdown tables. Scripts live in the skill
  directory and reuse rocprofv3 directly without any project-specific
  assumptions. Usage: /kernel-perf-analysis
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
│   ├── perf-benchmarking.md    # Agent: timing + MFMA efficiency table
│   ├── perf-counter.md         # Agent: hardware counter collection
│   ├── perf-trace-analysis.md  # Agent: ATT trace + MFMA loop analysis
│   └── att-runner.md           # Legacy low-level runner (kept for reference)
├── scripts/
│   ├── run_perf_table.py         # Mode 1: parse kernel_trace.csv → perf table
│   ├── run_counter_collection.py # Mode 2: parse counter_collection.csv → table
│   ├── run_att.py                # Mode 3: run process_json.py on ui_* dir
│   └── process_json.py           # ATT post-processor: MFMA efficiency, timing
└── references/
    └── lds-analysis-and-optimization.md  # LDS throughput & layout reference
```

---

## Step 0 — Determine which modes to run

### Mode selection logic

| User mentions | Modes to run |
|---------------|-------------|
| "how fast", "benchmark", "VGPR", "perf table", "compare" | Mode 1 |
| "counter", "bank conflict", "cache", "PMC", any counter name | Mode 2 |
| "trace", "ATT", "MFMA efficiency", "bottleneck", "lgkmcnt", "vmcnt" | Mode 3 |
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

## Step 5 — Comprehensive summary and suggestions

After presenting all individual results, produce a `## Summary and
Optimization Suggestions` section. Ground every suggestion in the reference
document `references/lds-analysis-and-optimization.md`.

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

### 5.4 Loop structure (from Mode 3)

- `pro_ratio` > 10% → prologue is too long; check if async DMA warmup is
  amortized correctly
- `epi_ratio` > 5% → epilogue tail latency; consider `/gluon-beyond-loop-opt`
- `loop_ratio` < 85% → significant time outside the K-loop

### 5.5 Summary format

```
## Summary and Optimization Suggestions

**Overall assessment:** <one sentence>

| Metric | Value | Status |
|--------|-------|--------|
| MFMA efficiency | <value> | ✅ / ⚠ / ❌ |
| LDS bank conflicts | <count/dispatch> | ✅ / ⚠ / ❌ |
| avg kernel time | <value> | — |
| loop ratio | <value> | ✅ / ⚠ |

**Recommended next steps (in priority order):**
1. <step 1 with skill name and rationale grounded in reference>
2. <step 2 ...>
...

**Reference:** See `references/lds-analysis-and-optimization.md` for
the LDS throughput model and layout strategy guidance used above.
```

---

## Mode Selection Reference

| Counter | Measures | Threshold |
|---------|---------|-----------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflicts | 0 = clean; > 0 = conflicts |
| `SQ_LDS_DATA_FIFO_FULL` | LDS data FIFO saturation | Should be 0 |
| `TCC_EA0_RDREQ_DRAM_sum` | L2→DRAM read requests | High = HBM-bound |
| `TCP_TCC_READ_REQ_sum` | L1→L2 read requests (cache misses) | High = L1 pressure |
| `GRBM_GUI_ACTIVE` | GPU active cycles | — |
