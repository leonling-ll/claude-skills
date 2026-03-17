---
name: kernel-perf-analysis
description: >
  Collect AMD GPU kernel performance metrics for any Python kernel file using
  rocprofv3. Three modes auto-selected based on user intent: (1) general
  performance table — TFLOPS, VGPRs, spill count, MFMA efficiency, average
  kernel time — triggered by "how fast is", "benchmark", "TFLOPS", "VGPR",
  "perf table", or "compare versions"; (2) hardware performance counter
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

Collect AMD GPU kernel performance data for any Python kernel file.
Each mode is split into two decoupled responsibilities:

| Responsibility | Who does it | Tool |
|----------------|-------------|------|
| **Run rocprofv3** | `att-runner` agent | Agent tool |
| **Parse & format results** | analysis scripts in `scripts/` | Bash tool |

---

## Directory Structure

```
kernel-perf-analysis/
├── SKILL.md
├── agents/
│   └── att-runner.md             # Agent: runs rocprofv3, returns output paths
└── scripts/
    ├── run_perf_table.py         # Mode 1: parse kernel_trace.csv → perf table
    ├── run_counter_collection.py # Mode 2: parse counter_collection.csv → table
    ├── run_att.py                # Mode 3: run process_json.py on ui_* dir
    └── process_json.py           # ATT post-processor: MFMA efficiency, loop timing
```

---

## The att-runner Agent

The `att-runner` agent handles **all rocprofv3 invocations**. Spawn it via the
Agent tool with subagent_type `general-purpose` and the contents of
`agents/att-runner.md` as instructions. It:

1. Validates rocprofv3 is available
2. Writes any required config files (YAML counters, JSON ATT config)
3. Runs rocprofv3 with the correct flags for the requested mode
4. Returns structured output: `csv_path` (modes 1 & 2) or `ui_dir` (mode 3)

**Agent modes:**
- `kernel-trace` → timing CSV for Mode 1
- `counter`      → counter CSV for Mode 2
- `att`          → ui_* directory for Mode 3

---

## Mode 1: General Performance Table

### Step 1 — Collect in parallel (two att-runner agents)

Spawn **two att-runner agents in parallel** (single Agent tool message with two
calls) so timing and MFMA efficiency are collected simultaneously:

**Agent A — kernel-trace:**
```
Kernel file: <absolute path>
Mode: kernel-trace
Options:
  kernel_name: "<regex matching kernel function name>"
  iters: 20
Output directory: /tmp/kperf_trace_<label>
```
Returns: `csv_path = /tmp/kperf_trace_<label>/hostname/pid_kernel_trace.csv`

**Agent B — att:**
```
Kernel file: <absolute path>
Mode: att
Options:
  kernel_name: "<regex matching kernel function name>"
  att_lib: "/var/lib/jenkins/att-decoder-v3-3.0.0-Linux/opt/rocm/lib/"
Output directory: /tmp/kperf_att_<label>
```
Returns: `ui_dir = /tmp/kperf_att_<label>/ui_<kernel_name>`

### Step 2 — Analyze ATT first (run_att.py), then perf table

**2a.** Extract MFMA efficiency from the ATT output:
```bash
SKILL=/home/leling/claude-skills/kernel-perf-analysis

MFMA_EFF=$(python3 $SKILL/scripts/run_att.py --ui-dir <ui_dir> \
  | python3 -c "import sys,re; m=re.search(r'\"mfma efficiency\".*?\"([\d.]+%)\"', sys.stdin.read()); print(m.group(1) if m else '')")
```

**2b.** Compute TFLOPS from the known problem dimensions, then print table:
```bash
AVG_US=$(python3 $SKILL/scripts/run_perf_table.py \
  --csv <csv_path> --kernel-file <kernel.py> \
  --kernel-name "<kernel_name>" --iters 20 --label "tmp" \
  | grep "dispatches found" | grep -oP '[\d.]+ µs' | head -1 | tr -d ' µs')

TFLOPS=$(python3 -c "print(f'{2*B*M*K*N / (${AVG_US}e-6) / 1e12:.1f}')")

python3 $SKILL/scripts/run_perf_table.py \
  --csv <csv_path> \
  --kernel-file <kernel.py> \
  --kernel-name "<kernel_name>" \
  --iters 20 \
  --tflops $TFLOPS \
  --mfma-eff "$MFMA_EFF" \
  --label "<label>"
```

**Output:**
```
| Version              | TFLOPS | VGPRs | Spills | MFMA Eff. | avg time  |
|----------------------|--------|-------|--------|-----------|-----------|
| my_label             |    118 |   200 |      0 |    57.98% | 795.88 us |
```

**TFLOPS note:** `run_perf_table.py` does not compute TFLOPS automatically
(it doesn't know M/N/K/B). Read the problem dimensions from the kernel file,
then compute: `2 * B * M * K * N / (avg_us * 1e-6) / 1e12`.

**Practical approach:** Parse `avg_us` from the script's `  N dispatches found,
avg of last 20 = <avg_us> µs` line, compute TFLOPS in Python, then re-run with
`--tflops` and `--mfma-eff`.

### Options

| Option | Description |
|--------|-------------|
| `--csv PATH` | Path to *_kernel_trace.csv from att-runner (required) |
| `--kernel-file PATH` | Original .py file for Triton cache lookup |
| `--kernel-name STR` | Substring/regex matching kernel name in CSV |
| `--iters N` | Last N dispatches to average (default: 20) |
| `--tflops FLOAT` | Pre-computed TFLOPS (optional) |
| `--mfma-eff STR` | MFMA efficiency e.g. "57.98%" (optional) |
| `--label STR` | Version column label |

---

## Mode 2: Hardware Counter Collection

### Step 1 — Collect (att-runner agent, mode: counter)

Spawn the att-runner agent:
```
Kernel file: <absolute path>
Mode: counter
Options:
  counters: ["SQ_LDS_BANK_CONFLICT", "SQ_LDS_DATA_FIFO_FULL"]
  kernel_name: "<optional regex>"
Output directory: <temp dir>
```

Agent returns: `csv_path = /tmp/.../hostname/pid_counter_collection.csv`

### Step 2 — Analyze (run_counter_collection.py)

```bash
python3 $SKILL/scripts/run_counter_collection.py \
  --csv <csv_path> \
  --counters SQ_LDS_BANK_CONFLICT,SQ_LDS_DATA_FIFO_FULL \
  --kernel-name "<kernel_name>" \
  --label "my_label"
```

**Output:**
```
| Version              | SQ_LDS_BANK_CONFLICT          | SQ_LDS_DATA_FIFO_FULL         | Dispatches |
|----------------------|-------------------------------|-------------------------------|------------|
| my_label             |                    12,345,678 |                             0 |         20 |
```

### Options

| Option | Description |
|--------|-------------|
| `--csv PATH` | Path to *_counter_collection.csv from att-runner (required) |
| `--counters A,B,...` | Comma-separated counter names to display (required) |
| `--kernel-name STR` | Kernel name substring for filtering rows |
| `--label STR` | Version column label |

### Common counters

| Counter | Measures |
|---------|---------|
| `SQ_LDS_BANK_CONFLICT` | LDS bank conflicts (high = use swizzle/padding) |
| `SQ_LDS_DATA_FIFO_FULL` | LDS data FIFO saturation |
| `TCC_EA0_RDREQ_DRAM_sum` | L2→DRAM read requests (HBM traffic) |
| `TCP_TCC_READ_REQ_sum` | L1→L2 read requests (cache misses) |
| `GRBM_GUI_ACTIVE` | GPU active cycles |

---

## Mode 3: ATT Trace + MFMA Efficiency Analysis

### Step 1 — Collect (att-runner agent, mode: att)

Spawn the att-runner agent:
```
Kernel file: <absolute path>
Mode: att
Options:
  kernel_name: "<regex matching kernel function name>"
  att_lib: "/var/lib/jenkins/att-decoder-v3-3.0.0-Linux/opt/rocm/lib/"
Output directory: <temp dir or persistent path>
```

Agent returns: `ui_dir = /tmp/.../ui_matmul_kernel`

### Step 2 — Analyze (run_att.py)

```bash
python3 $SKILL/scripts/run_att.py --ui-dir <ui_dir>
```

**Output:**
```json
{
  "loop_first_index": 1234,
  "mfma_count_in_loop": 16,
  "total_mfma_cycles_in_loop": 256,
  "num_iterations": 64.0,
  "average_loop_duration": 15620.5,
  "average_prologue_duration": 342.1,
  "average_epilogue_duration": 128.4,
  "pro_ratio": "2.10%",
  "loop_ratio": "96.01%",
  "epi_ratio": "0.79%",
  "average_iteration_duration": 244.07,
  "mfma efficiency": "57.98%"
}

--- ATT Analysis Summary ---
  MFMA efficiency      : 57.98%
  Avg iteration cycles : 244.1
  Loop iterations      : 64.0
  Time distribution    : prologue=2.10%, loop=96.01%, epilogue=0.79%
  ⚠ MFMA efficiency < 60%: kernel is memory-bound.
```

### Options

| Option | Description |
|--------|-------------|
| `--ui-dir PATH` | Path to ui_* directory from att-runner (required) |

### Interpreting results

| Field | Meaning |
|-------|---------|
| `mfma efficiency` | MFMA cycles / total iteration cycles; target > 80% |
| `average_iteration_duration` | Cycles per K-loop iteration |
| `pro_ratio` | % of total time in prologue (should be small) |
| `loop_ratio` | % of total time in main loop (should be > 90%) |

**< 60%:** Memory-bound. Apply `/gluon-pipeline-opt` or `/gluon-lds-opt`.
**60–80%:** Partially compute-bound. Some pipeline improvement possible.
**> 80%:** Compute-bound. Kernel is well-optimized.

---

## Workflow: Full Analysis Pipeline

```
SKILL=/home/leling/claude-skills/kernel-perf-analysis
KERNEL=<absolute path to kernel.py>

# Step 1: Spawn att-runner (kernel-trace mode), get csv_path
# Step 2: python3 $SKILL/scripts/run_perf_table.py --csv <csv_path> ...

# Step 3: Spawn att-runner (counter mode), get csv_path
# Step 4: python3 $SKILL/scripts/run_counter_collection.py --csv <csv_path> ...

# Step 5: Spawn att-runner (att mode), get ui_dir
# Step 6: python3 $SKILL/scripts/run_att.py --ui-dir <ui_dir>
```

---

## Mode Selection Logic

1. Counter names / bank conflicts / cache stats mentioned → **Mode 2**
2. Trace / MFMA efficiency / bottleneck / ATT / lgkmcnt / vmcnt → **Mode 3**
3. Otherwise (benchmark / TFLOPS / VGPRs / speed / how fast) → **Mode 1**

Run multiple modes when the request covers multiple concerns (e.g., "benchmark
and check bank conflicts" → Mode 1 + Mode 2, spawning att-runner twice in
parallel for the two collect steps).
