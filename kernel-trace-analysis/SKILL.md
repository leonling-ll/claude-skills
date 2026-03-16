---
name: kernel-trace-analysis
description: >
  Profile GPU kernels using rocprofv3 to collect kernel stats and instruction-level
  traces, then analyze the trace data to identify performance bottlenecks (barrier stalls,
  idle cycles, TA-blocked loads) and produce an optimization plan. Workflow: (1) run
  rocprofv3 --stats to discover kernel names, (2) configure input.yaml with the target
  kernel regex, (3) run rocprofv3 -i input.yaml to collect ATT traces, (4) parse the
  CSV trace output to find high-cycle instructions and stall reasons, (5) produce a
  bottleneck report with actionable optimization suggestions.
  Usage: /kernel-trace-analysis <cmd>
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Kernel Trace Analysis

Profile and analyze GPU kernel instruction traces using `rocprofv3` to identify
performance bottlenecks and produce an optimization plan.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<CMD>` | Yes | The full command to profile. Example: `python bench_pa.py --batch 32` |

When this skill is invoked, the argument is the command to profile. Replace `<CMD>`
below with the provided command. If no command is provided, ask the user for it.

## Prerequisites

- `rocprofv3` must be available in PATH (comes with ROCm 6.x+)
- The input.yaml template at `~/Documents/input.yaml` (will be copied and modified)
- Sufficient disk space for trace output (ATT traces can be large)
- **computer-use MCP server** must be installed for GUI trace visualization with `rocprof-compute-viewer`

### Install computer-use MCP

Before running this skill, ensure the official computer-use MCP server is configured.
Run the following command (only needed once):

```bash
claude mcp add --scope user --transport stdio computer-use -- npx -y computer-use-mcp
```

This enables Claude Code to take screenshots, click, and type in GUI applications
(needed to operate `rocprof-compute-viewer.exe` via Wine for visual trace inspection).

Also ensure system dependencies are installed:

```bash
sudo apt-get install -y xdotool xclip imagemagick
```

### rocprof-compute-viewer Setup

ROCprof Compute Viewer (RCV) visualizes ATT trace data from `ui_output_agent_*_dispatch_*` directories.
It is a Windows Qt6 application run via Wine:

```
~/.wine/drive_c/Program Files (x86)/ROCprof-Compute-Viewer/bin/rocprof-compute-viewer.exe
```

Launch with a `ui_output_agent_XXX_dispatch_YYY` directory as argument:

```bash
DISPLAY=:0 wine "C:\\Program Files (x86)\\ROCprof-Compute-Viewer\\bin\\rocprof-compute-viewer.exe" \
  "Z:\\path\\to\\ui_output_agent_XXX_dispatch_YYY" &
```

**Important**: The MCP computer-use screenshot tool may not work with multi-monitor setups
(Wine windows often open on the secondary monitor). Use `xwd` + `imagemagick` instead:

```bash
# Find the main viewer window (look for "ROCprof Compute Viewer" title)
DISPLAY=:0 xdotool search --name "ROCprof Compute Viewer"
# Move it to primary display if needed
DISPLAY=:0 xdotool windowmove <WINDOW_ID> 50 50
# Capture the window
DISPLAY=:0 xwd -id <WINDOW_ID> | convert xwd:- /tmp/rocprof_screenshot.png
```

Use the `Read` tool to view the resulting PNG (Claude Code is multimodal).

### RCV Keyboard Shortcuts & Mouse Controls

**Plots (Wave States, Occupancy, Dispatches):**
| Action | Control |
|--------|---------|
| Zoom horizontal | Mouse wheel |
| Zoom vertical | Ctrl + Mouse wheel |
| Pan | Right click + drag |
| Reset axis | Ctrl + Left click |
| Select area | Left click + drag |
| Measure cycles | Right click + drag (also in Global View, CU, Utilization) |

**Compute Unit & Utilization tabs:**
| Action | Control |
|--------|---------|
| Pan left/right | A / D keys |
| Vertical scroll | Mouse wheel |
| Zoom in/out | Ctrl + Mouse wheel |
| Highlight ISA | Left click on token |

**Tab navigation:**
| Action | Control |
|--------|---------|
| Switch tab | Left click on tab header |
| Keep multiple tabs open | Ctrl + Left click on tab header |

### RCV Tab Reference

| Tab | Shows | Key Use |
|-----|-------|---------|
| **Hotspot** | Histogram of instruction latency costs (stall+execute, no idle). Bins adjustable via Edit→Hotspot Options. Click a bin to highlight ISA lines in it. Computed over WaveView Clock Range. |
| **Instructions** | ISA list with Hitcount/Latency per instruction. Cost mode: mean/sum per wave or all waves. Arrows link memory ops to their s_waitcnt. Left/right arrows navigate to SQTT token. Hover/click ISA ↔ source line. |
| **Wave States** | Active waves per state (IDLE, EXEC, STALL, WAIT) over time for target CU. Vertical slice of CU tab. |
| **Occupancy** | Waves per Shader Engine over time. |
| **Kernel Dispatches** | Per-kernel occupancy timeline. |
| **Compute Unit** | Per-wave trace timeline (one row per SIMD-Slot). Color = instruction type. |
| **Utilization** | Per-SIMD trace by instruction type (VALU, VMEM, SCALAR, OTHER). Hides IMMED tokens and stalled time, shows only issue/execution. Good for finding pipeline bubbles. |
| **Counters** | Hardware perf counter plots (if collected with `--att-perfcounters`). |
| **Global View** | All waves across all Shader Engines, color-coded by kernel. |
| **Summary** | (MI2xx/MI3xx only) Average instruction cost, hardware utilization by type, per-CU rates. Requires `--att-activity 10` or explicit SQ_ACTIVE_INST_* counters. |
| **Explorer** | Hierarchical file browser with per-file hotspot bars. Click file to see top latency lines. |

### RCV Left Side Panel

- **Shader/SIMD/Slot/WaveID**: Select target wave for instruction-level analysis
- **WaveView Clock Range**: Defines visible cycles for CU/Utilization tabs and Hotspot calculation. Narrow it to focus on specific regions.
- **GlobalView zoom** [0-15]: Zoom level for Global View
- **WaveView zoom** [0-10]: Zoom level for trace views
- **Iteration**: Loop iteration navigator. Click a token to set, or edit directly to jump between loop iterations.
- **Search**: Find text in ISA view (e.g., `ds_` to find LDS instructions, `mfma` to find MFMA)
- **History**: Go back to previously selected tokens

### Collecting Summary Data

To enable the Summary tab (MI300X), collect with activity counters:

```bash
# Convenience parameter
rocprofv3 --att-activity 10 --att ...

# Or explicit counters
rocprofv3 --att-perfcounter-ctrl 10 \
  --att-perfcounters "SQ_BUSY_CU_CYCLES SQ_VALU_MFMA_BUSY_CYCLES SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_FLAT SQ_ACTIVE_INST_SCA SQ_ACTIVE_INST_MISC" \
  --att ...
```

### Collecting Hardware Counters

Up to 8 SQ counters (4 recommended). On MI300, `--att-perfcounter-ctrl 3` gives 120-240 cycle polling:

```bash
rocprofv3 --att-perfcounter-ctrl 3 \
  --att-perfcounters "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INST_LEVEL_LDS" \
  --att ...
```

Per-SIMD filtering with `:0xMask` (default 0xF = all SIMDs):
```bash
--att-perfcounters "SQ_INSTS_VALU:0xF SQ_INSTS_SALU:0x1 SQ_INSTS_SALU:0x2"
```

## Workflow Overview

```
Step 1: Kernel Discovery      rocprofv3 --stats --kernel-trace
Step 2: Configure Tracing     Edit input.yaml with target kernel regex
Step 3: Collect ATT Trace     rocprofv3 -i input.yaml -- <cmd>
Step 4: Parse Trace Data      Read CSV files from ui_output_agent_xxx/
Step 5: Analyze Bottlenecks   Identify barrier, idle, TA stalls
Step 6: Report                Produce optimization plan
```

---

## Step 1: Kernel Discovery

Run `rocprofv3` in stats mode to list all GPU kernels launched by the command,
sorted by total duration:

```bash
rocprofv3 --stats --kernel-trace -f csv -- <CMD> 2>&1
```

This produces a `kernel_trace_stats.csv` (or similar) file. Parse it to find:
- Kernel names matching the user's area of interest
- Total duration, call count, and average duration per kernel
- The hottest kernels (most total GPU time)

Present the kernel list to the user in a table format:

| Rank | Kernel Name | Calls | Total (us) | Avg (us) | % GPU Time |
|------|-------------|-------|------------|----------|------------|

Ask the user which kernel(s) to trace if not obvious. If the user already
specified a kernel name pattern, proceed directly.

**Important**: The `--stats` output directory and file naming varies by ROCm version.
Look for CSV files in the current directory or any newly created output directories:

```bash
# Find the stats output
find . -maxdepth 3 -name "*stats*" -newer /tmp/trace_timestamp -type f 2>/dev/null
```

Create a timestamp file before profiling to identify new output:

```bash
touch /tmp/trace_timestamp
```

### 1.1 Using the rocprofv3 SQLite Database (Preferred)

`rocprofv3 --kernel-trace` also produces a `results.db` SQLite database. This is often more reliable than parsing CSV files and provides structured access to kernel timing AND VGPR counts:

```bash
# Find the database
find . -maxdepth 3 -name "results.db" -newer /tmp/trace_timestamp 2>/dev/null
```

```sql
-- Kernel timing summary
SELECT ks.KernelName,
       COUNT(*) as calls,
       ROUND(AVG(kd.end - kd.start)/1000.0, 1) as avg_us,
       ROUND(MIN(kd.end - kd.start)/1000.0, 1) as min_us,
       ROUND(MAX(kd.end - kd.start)/1000.0, 1) as max_us
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
GROUP BY ks.KernelName
ORDER BY avg_us DESC
LIMIT 20;

-- VGPR allocation (arch vs accum - critical for CDNA3 occupancy analysis)
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count,
       ki.sgpr_count, ki.lds_size
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
WHERE ks.KernelName LIKE '%target_kernel%'
LIMIT 5;
```

### 1.2 GPU Kernel Time vs End-to-End Time (CRITICAL)

**Always measure GPU kernel time via rocprofv3, not end-to-end wall-clock time.**

For Triton/Gluon kernels, the CPU-side JIT dispatch overhead can be 100-200+ us per call (argument specialization, hash computation, grid calculation). For kernels with GPU time < 200us, this CPU overhead **dominates** end-to-end measurements like `triton.testing.do_bench`.

A kernel change that increases GPU time by 4.5x can appear neutral in end-to-end benchmarks because the CPU overhead masks the regression. Always compare GPU kernel duration from `rocprofv3 --kernel-trace` before and after optimization.

---

## Step 2: Configure input.yaml

Copy the template input.yaml and modify `kernel_include_regex` to match the
target kernel:

```bash
cp ~/Documents/input.yaml /tmp/trace_input.yaml
```

Edit `/tmp/trace_input.yaml` to set:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_NAME_PATTERN>
       kernel_iteration_range: "[1, [3-4]]"
       output_file: out
       output_directory: kernel_trace_output
       output_format: [json, csv, otf2, pftrace]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key configuration notes:
- `kernel_include_regex`: Use the exact kernel name or a regex pattern. For
  Triton kernels, names often contain the function name, e.g.,
  `paged_attention_decode_sliding_window` matches the sliding window PA kernel.
- `kernel_iteration_range`: `"[1, [3-4]]"` means skip the first dispatch (warmup)
  and trace dispatches 3 and 4. Adjust if the command does fewer iterations.
- `att_target_cu: 1`: Trace a single CU to keep output manageable.
- `att_shader_engine_mask: "0xf"`: Collect from 4 shader engines.
- `att_simd_select: "0xf"`: Collect all 4 SIMDs on the selected CU.
- `att_buffer_size`: 96MB per SE. Increase if trace is truncated.

---

## Step 3: Collect ATT Trace

### Option A: Using input.yaml (recommended for full control)

```bash
rocprofv3 -i /tmp/trace_input.yaml -- <CMD> 2>&1
```

### Option B: Using CLI flags (simpler, no yaml needed)

```bash
rocprofv3 --att \
  --att-target-cu 1 \
  --att-shader-engine-mask 0xf \
  --att-simd-select 0xf \
  --att-buffer-size 0x6000000 \
  --kernel-include-regex "<KERNEL_NAME_PATTERN>" \
  -o /path/to/output \
  -- <CMD> 2>&1
```

**Note**: `--att` is a boolean flag (no value). The ATT library `librocprof-trace-decoder.so`
must be in `/opt/rocm/lib/`. If missing, install from
[rocprof-trace-decoder releases](https://github.com/ROCm/rocprof-trace-decoder/releases):

```bash
wget -q https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
chmod +x rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=/tmp/rtd-install
find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \;
ldconfig
```

### Output Structure

Both options create a directory containing:

```
output_directory/
├── ui_output_agent_<PID>_dispatch_<N>/    # Per-dispatch trace (RCV compatible)
│   ├── code.json                          # Per-instruction stall/idle/latency
│   ├── occupancy.json                     # Occupancy data
│   ├── filenames.json                     # Source file mapping
│   ├── snapshots.json                     # Trace snapshots
│   ├── wstates*.json                      # Wave state data
│   ├── se*_sm*_sl*_wv*.json              # Per-wave raw traces
│   ├── source_0_*.py                      # Snapshotted source files
│   └── source_1_*.py
├── stats_ui_output_agent_*_dispatch_*.csv # Per-dispatch stats
├── out_kernel_trace.csv                   # Kernel launch info
├── out_*_shader_engine_*.att              # Raw ATT data per SE
├── out_gfx942_code_object_id_*.out        # Code object binaries
├── out_results.json                       # Full results (can be large, 100MB+)
├── out_hip_api_trace.csv                  # HIP API trace
└── out_hsa_api_trace.csv                  # HSA API trace
```

The `ui_output_agent_*_dispatch_*` directories are what rocprof-compute-viewer opens.

Locate the trace output:

```bash
find . -type d -name "ui_output_agent_*" -newer /tmp/trace_timestamp 2>/dev/null
```

---

## Step 4: Parse Trace Data

Read the CSV trace files and extract instruction-level timing data.

### 4.1 Identify the CSV format

The ATT trace CSV format may vary. Common column layouts:

**Format A** (per-instruction):
```
pc,instruction,type,cycles,comment
```

**Format B** (aggregated):
```
instruction,count,total_cycles,avg_cycles,category
```

Read the CSV header first to determine the format, then parse accordingly.

### 4.2 Extract high-cycle instructions

The trace output may be in **JSON** or **CSV** format depending on the `output_format` config.

#### JSON format (rocprof-compute-viewer compatible)

When `output_format` includes `json`, the key file is `code.json`:

```python
# Parse code.json
import json
with open('ui_output_agent_XXX_dispatch_YYY/code.json') as f:
    data = json.load(f)

header = data['header']  # "ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle"
instructions = data['code']
# Each instruction: [isa_string, _, line_num, source_file:line, codeobj, vaddr, hit, latency, stall, idle]
# Index:             0          1  2         3                  4       5      6    7        8      9

# Top instructions by stall
hot = sorted([i for i in instructions if i[8] > 0], key=lambda x: x[8], reverse=True)
for inst in hot[:20]:
    print(f"L{inst[2]:>4d}  stall={inst[8]:>6d}  idle={inst[9]:>6d}  {inst[0][:60]}  | {inst[3]}")
```

Also check:
- `realtime.json` — per-SE clock offsets: `{"SE0": [[gfx_clock, realtime_clock], ...], ...}`
- `se*_sm*_sl*_wv*.json` — per-wave raw trace data
- `filenames.json` — source file mapping
- `occupancy.json` — occupancy data

#### CSV format

```bash
# Check the header
head -1 kernel_trace_output/ui_output_agent_*/out*.csv
# Sort by cycles (adjust column number as needed)
sort -t, -k4 -nr kernel_trace_output/ui_output_agent_*/out*.csv | head -30
```

### 4.3 Aggregate analysis

Use Python for comprehensive analysis. Key aggregations:

```python
# 1. Instruction type distribution (by total stall)
from collections import defaultdict
type_stats = defaultdict(lambda: {'count': 0, 'latency': 0, 'stall': 0, 'idle': 0})
for inst in instructions:
    op = inst[0].split()[0]
    type_stats[op]['count'] += 1
    type_stats[op]['stall'] += inst[8]
    type_stats[op]['idle'] += inst[9]

# 2. Source line stall breakdown
line_stats = defaultdict(lambda: {'stall': 0, 'barrier': 0, 'waitcnt': 0, 'mfma': 0})
for inst in instructions:
    if inst[3]:
        line = inst[3].split(':')[-1]
        line_stats[line]['stall'] += inst[8]
        if 's_barrier' in inst[0]: line_stats[line]['barrier'] += 1
        if 's_waitcnt' in inst[0]: line_stats[line]['waitcnt'] += 1
        if 'v_mfma' in inst[0]: line_stats[line]['mfma'] += 1

# 3. Register usage estimation
import re
max_vgpr = max(int(m.group(1)) for inst in instructions for m in re.finditer(r'v\[?(\d+)', inst[0]))
max_agpr = max((int(m.group(1)) for inst in instructions for m in re.finditer(r'a\[?(\d+)', inst[0])), default=0)
print(f"VGPR: {max_vgpr+1}, AGPR: {max_agpr+1}, Total: {max_vgpr+max_agpr+2}/512")
```

### 4.4 Visual Analysis with rocprof-compute-viewer

After programmatic analysis, open the trace in RCV for visual inspection:

```bash
DISPLAY=:0 wine "C:\\Program Files (x86)\\ROCprof-Compute-Viewer\\bin\\rocprof-compute-viewer.exe" \
  "Z:\\path\\to\\ui_output_agent_XXX_dispatch_YYY" &
sleep 5
# Find and capture the window
WID=$(DISPLAY=:0 xdotool search --name "ROCprof Compute Viewer" | head -1)
DISPLAY=:0 xdotool windowmove $WID 50 50  # Move to primary display
DISPLAY=:0 xwd -id $WID | convert xwd:- /tmp/rcv_overview.png
```

**Visual analysis checklist:**

1. **Hotspot tab**: Check the histogram for dominant instruction bins. Click bins to jump to ISA.
2. **Instructions tab**: Sort by "Latency: Sum all" to see aggregate costs. Use arrows to trace memory→waitcnt dependencies.
3. **Compute Unit tab**: Look for waves with long stall bars (yellow/red). Use A/D to pan, Ctrl+wheel to zoom.
4. **Utilization tab**: Check for pipeline bubbles (gaps between VALU/VMEM bars). If MFMA bars are sparse, the kernel is memory-bound.
5. **Wave States tab**: Check STALL vs EXEC ratio. High STALL = memory-bound.
6. **Summary tab** (if collected with `--att-activity`): Read hardware utilization percentages directly.

**Capture specific tabs:**
```bash
# Click tab with xdotool (use coordinates relative to window)
DISPLAY=:0 xdotool mousemove --window $WID <TAB_X> <TAB_Y> && xdotool click 1
sleep 1
DISPLAY=:0 xwd -id $WID | convert xwd:- /tmp/rcv_<tab_name>.png
```

### 4.5 Categorize bottleneck instructions

Group the high-cycle instructions into categories:

| Category | Typical Instructions | Root Cause |
|----------|---------------------|------------|
| **Barrier/Wait** | `s_waitcnt vmcnt(0)`, `s_barrier`, `s_waitcnt lgkmcnt(0)` | Waiting for memory or synchronization |
| **Idle** | `s_nop`, `s_sleep`, idle cycles between instructions | Pipeline bubbles, underutilization |
| **VMEM Load** | `buffer_load_dword`, `buffer_load_b128`, `global_load_*` | TA (Texture Addresser) blocking, cache misses |
| **SMEM Load** | `s_load_dword*` | Scalar memory stalls |
| **MFMA** | `v_mfma_f32_*` | Matrix compute (expected to be high if compute-bound) |
| **LDS** | `ds_read_*`, `ds_write_*`, `ds_bpermute_*`, `ds_swizzle_*` | Shared memory bank conflicts or cross-lane reduce |

---

## Step 5: Analyze Bottlenecks

### 5.1 Barrier Stall Analysis

High-cycle `s_waitcnt` and `s_barrier` instructions indicate the kernel is
**memory-bound** -- the wavefront is stalling because data hasn't arrived yet.

Check what the barrier is waiting for:
- `s_waitcnt vmcnt(N)`: Waiting for N outstanding VMEM (global memory) loads to complete.
  If `vmcnt(0)`, waiting for ALL loads. Root cause: load latency not hidden.
- `s_waitcnt lgkmcnt(N)`: Waiting for LDS (shared memory) or SMEM operations.
  Root cause: LDS bank conflicts or SMEM stalls.
- `s_barrier`: Cross-wave synchronization. All waves in the workgroup must reach
  this point. Root cause: workload imbalance between waves.

**Optimization hints for barrier stalls**:
1. **Prefetch / double-buffer**: Issue loads earlier to overlap with compute
   (see `/prefetch-data-load` skill)
2. **Reduce waitcnt strictness**: Change `vmcnt(0)` to `vmcnt(N)` where N > 0
   if not all loads need to complete before proceeding
3. **Increase occupancy**: More waves in flight hides memory latency
4. **Restructure loop**: Move loads before compute in the loop body

### 5.2 Idle Cycle Analysis

Idle cycles (gaps between instructions, `s_nop`) indicate:
- **Pipeline bubbles**: Dependent instructions too close together
- **MFMA drain**: Waiting for matrix operations to complete
- **Register dependency**: Next instruction depends on result of previous

**Optimization hints for idle cycles**:
1. **Interleave independent operations**: Put unrelated ALU work between
   dependent instructions
2. **Software pipelining**: Overlap iterations to fill idle slots
3. **Adjust MFMA shape**: Smaller MFMA instructions have shorter latency but
   lower throughput. Choose based on whether latency or throughput matters.

### 5.3 TA-Blocked Load Analysis

When `buffer_load_*` or `global_load_*` instructions show high cycle counts,
the Texture Addresser (TA) unit is likely blocked:

- **TA queue full**: Too many outstanding loads. Reduce the number of concurrent
  loads or increase the gap between them.
- **Cache thrashing**: Access pattern causes L2 cache evictions. Check for
  non-sequential or strided access patterns.
- **TLB misses**: Large working set causes page table walks. Ensure data locality.
- **Cross-CU contention**: Multiple CUs competing for the same memory channels.

**Optimization hints for TA-blocked loads**:
1. **Coalesce loads**: Ensure threads access consecutive addresses to maximize
   cache line utilization (128 bytes per transaction on CDNA3)
2. **Vectorize loads**: Use `buffer_load_b128` (16 bytes) instead of multiple
   smaller loads
3. **Reduce load count in flight**: If TA is saturated, issue fewer concurrent loads
4. **Prefetch to LDS**: Load to shared memory first, then read from shared memory
   to avoid repeated global loads

### 5.4 Cross-Wave Reduce Stall Analysis (Gluon Kernels)

Gluon kernels (e.g., paged attention) often have cross-wave reduce phases (softmax max/sum) that use `ds_bpermute` → `ds_swizzle` → `ds_write LDS` → `s_barrier` → `ds_read LDS` patterns. These can dominate stall time.

**Identification**: Look for clusters of `s_barrier` + `s_waitcnt lgkmcnt(0)` + `ds_*` instructions from the same source line (e.g., `:189` for softmax reduce). If the total stall in this region exceeds 30% of kernel stalls, it's a primary bottleneck.

**Analysis approach**:
1. Count barriers in the reduce region — more than 4 suggests the reduce can be optimized
2. Check if max and sum reduce are done in separate passes (doubles barrier count)
3. Look for `ds_write → barrier → ds_read` round-trips that could be replaced by direct `ds_bpermute`
4. Check if useful loads (e.g., next-phase data) can be hoisted into the barrier-wait dead time

**Key insight for Gluon**: You cannot eliminate `s_waitcnt` directly (compiler-inserted), but you CAN restructure the Python-level Gluon code to:
- Merge reduce passes (max + sum → single pass)
- Replace LDS write/read round-trips with `ds_bpermute` chains
- Issue next-phase loads before the reduce barriers
- See `/optimize-pa-decode-gluon` skill for specific strategies

### 5.5 Register Pressure Check

For gfx942 (MI308/MI300X), VGPRs are split into **two separate register files**:

```
arch_vgpr: General-purpose vector registers (VALU, VMEM, LDS ops)
accum_vgpr (AGPR): Accumulator registers (MFMA result writeback only)

Each SIMD has 256 arch_vgpr and 256 accum_vgpr slots.
Occupancy = 256 / max(arch_vgpr, accum_vgpr) waves per SIMD.

Examples:
  arch=148, accum=148 → 256/148 = 1 wave/SIMD (max of either = 148)
  arch=128, accum=148 → 256/148 = 1 wave/SIMD (accum is bottleneck)
  arch=128, accum=0   → 256/128 = 2 waves/SIMD (no accum)
  arch=256, accum=256  → 256/256 = 1 wave/SIMD
  > 256 in either file → SPILL (critical)
```

**IMPORTANT**: `maxnreg` can force `accum_vgpr=0`, which doubles occupancy but causes MFMA results to spill through arch_vgpr. This is almost always a net regression for MFMA-heavy kernels (measured 4.5x GPU slowdown). See `/optimize-pa-decode-gluon` Section 1.2.1 for details.

**How to check VGPR allocation** (from rocprofv3 database):
```sql
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
WHERE ks.KernelName LIKE '%target_kernel%'
LIMIT 5;
```

**From ISA trace** (less reliable, only shows registers actually used):
```python
import re
max_vgpr = max(int(m.group(1)) for inst in instructions for m in re.finditer(r'v\[?(\d+)', inst[0]))
max_agpr = max((int(m.group(1)) for inst in instructions for m in re.finditer(r'a\[?(\d+)', inst[0])), default=0)
print(f"arch_vgpr: ~{max_vgpr+1}, accum_vgpr: ~{max_agpr+1}")
print(f"Occupancy bottleneck: {'arch' if max_vgpr > max_agpr else 'accum'} ({max(max_vgpr, max_agpr)+1})")
```

Always report **both** arch_vgpr and accum_vgpr when analyzing traces. For prefetch feasibility, check headroom in the arch_vgpr file specifically (since prefetch buffers use arch_vgpr, not accum_vgpr).

### 5.6 MFMA Utilization

MFMA instructions should dominate cycle count in a well-optimized compute kernel.
Calculate MFMA utilization:

```
MFMA% = (total MFMA cycles) / (total kernel cycles) * 100
```

- **> 80%**: Compute-bound (good for compute-heavy kernels)
- **50-80%**: Mixed, some memory overhead
- **< 50%**: Memory-bound, significant optimization opportunity
- **< 20%**: Severely memory-bound, likely needs architectural changes

---

## Step 6: Generate Report

Produce a structured bottleneck report with the following sections:

### Report Template

```
# Kernel Trace Analysis Report

## Target Kernel
- **Name**: <kernel name>
- **Total Cycles**: <N>
- **MFMA Utilization**: <X>%

## Top Bottlenecks (by cycle cost)

### 1. <Category>: <instruction> — <cycles> cycles (<percentage>%)
- **Location**: PC offset 0x<addr>
- **Root Cause**: <explanation>
- **Impact**: <estimated cycle savings if fixed>
- **Suggested Fix**: <specific optimization>

### 2. ...

## Cycle Breakdown by Category
| Category | Total Cycles | % of Kernel | Assessment |
|----------|-------------|-------------|------------|
| MFMA     | ...         | ...         | compute    |
| Barrier  | ...         | ...         | stall      |
| VMEM     | ...         | ...         | load       |
| Idle     | ...         | ...         | waste      |
| SALU     | ...         | ...         | overhead   |
| LDS      | ...         | ...         | shared mem |

## Optimization Plan (Priority Order)

1. **[HIGH]** <optimization> — estimated <X>% cycle reduction
   - What: <describe the change>
   - Why: <link to bottleneck data>
   - How: <concrete implementation steps>

2. **[MEDIUM]** ...

3. **[LOW]** ...

## Recommended Next Steps
- [ ] Apply optimization #1 and re-profile
- [ ] Verify correctness after each change
- [ ] Compare cycle breakdown before/after
```

---

## Interpreting Common Patterns

### Pattern: s_waitcnt vmcnt(0) with high cycles before MFMA

```
buffer_load_b128 v[...], ...          ;  12 cycles
buffer_load_b128 v[...], ...          ;   8 cycles
s_waitcnt vmcnt(0)                    ; 847 cycles  <-- BOTTLENECK
v_mfma_f32_16x16x16_bf16 ...         ;  64 cycles
```

**Diagnosis**: Load latency is fully exposed. The wavefront issues loads and
then immediately waits for them, stalling for ~847 cycles.

**Fix**: Apply prefetch (double-buffer) to overlap loads with previous
iteration's MFMA compute. Expected to reduce wait cycles by 60-90%.

### Pattern: High-cycle buffer_load (TA blocked)

```
buffer_load_b128 v[...], ...          ; 523 cycles  <-- TA BLOCKED
buffer_load_b128 v[...], ...          ; 498 cycles  <-- TA BLOCKED
```

**Diagnosis**: TA unit is backed up. Too many concurrent loads or poor access
pattern causing cache misses.

**Fix**: (1) Reduce concurrent load count, (2) check access pattern for
coalescing, (3) consider loading to LDS with a single wavefront then broadcasting.

### Pattern: s_barrier with high cycles

```
s_barrier                             ; 1200 cycles  <-- BOTTLENECK
```

**Diagnosis**: Waves within the workgroup are not arriving at the barrier at the
same time. Some waves finish their work faster than others.

**Fix**: (1) Balance work across waves, (2) check if the barrier is necessary,
(3) reduce workgroup size if only a few waves are slow.

### Pattern: Idle/nop between MFMA instructions

```
v_mfma_f32_16x16x16_bf16 ...         ;  64 cycles
s_nop 7                              ;  32 cycles  <-- IDLE
s_nop 7                              ;  32 cycles  <-- IDLE
v_mfma_f32_16x16x16_bf16 ...         ;  64 cycles
```

**Diagnosis**: Pipeline bubbles between MFMA instructions. The second MFMA
depends on the result of the first, or there's insufficient interleaved work.

**Fix**: (1) Interleave VMEM loads or SALU work between MFMAs, (2) restructure
to allow independent MFMA chains, (3) consider using larger MFMA shapes to
reduce instruction count.

---

## FlyDSL Source-to-Assembly Mapping

FlyDSL kernels support source-to-assembly mapping in rocprofv3 ATT traces, enabling the same
`code.json` source annotations as Triton/Gluon kernels. This requires:

1. **`FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`** (default: true) — enables the `ensure-debug-info-scope-on-llvm-func` MLIR pass and `-g` flag
2. The pass converts MLIR `loc()` metadata into LLVM DWARF debug info (`DISubprogram`, `.debug_line`)
3. rocprofv3 reads `.debug_line` from the HSACO binary to produce source mappings

**Verification**: In `code.json`, each instruction entry has a source field (index 3):
```json
["s_load_dwordx8 s[8:15], s[0:1], 0x18", 0, 1, "/FlyDSL/kernels/pa_decode_fp8.py:123", ...]
```

**If source mappings are missing** (source field is empty `""`):
- Check that `FLYDSL_DEBUG_ENABLE_DEBUG_INFO` is not set to `0`/`false`
- Verify the `ensure-debug-info-scope-on-llvm-func` pass is in the pipeline (check `FLYDSL_DUMP_IR=1` output for stage `14_ensure_debug_info_scope_on_llvm_func`)
- Verify `.loc`/`.file` directives exist in `final_isa.s` (dump with `FLYDSL_DUMP_IR=1`)
- The pass must run AFTER `reconcile-unrealized-casts` and BEFORE `gpu-module-to-binary`

**How it compares to Triton**: Triton uses `passes.llvmir.add_di_scope(pm)` (a custom C++ pass in
`/triton/lib/Target/LLVMIR/LLVMDIScopePass.cpp`). FlyDSL uses MLIR's standard
`ensure-debug-info-scope-on-llvm-func` pass which is functionally equivalent.
Both convert MLIR source locations into LLVM `DISubprogramAttr` metadata.

## Error Handling

- If `rocprofv3` is not found: check `which rocprofv3` and suggest `export PATH=/opt/rocm/bin:$PATH`
- If no GPU is available: check `rocm-smi` output
- If trace output is empty: the kernel may not have been dispatched. Check kernel_include_regex.
- If trace is truncated: increase `att_buffer_size` (e.g., `"0xC000000"` for 192MB)
- If `kernel_iteration_range` doesn't match: the command may run fewer iterations. Try `"[0, [1-2]]"`
- If permission denied: `rocprofv3` may require `sudo` or the user to be in the `video` group
- If `rocprof-trace-decoder library path not found`: install the decoder library (see Step 3 above)
- If `--advanced-thread-trace <file>` fails with "invalid truth value": `--att` is a boolean flag, use `-i input.yaml` for file-based config or `--att` with separate `--att-*` CLI flags
- If `--stats` fails with "No tracing options enabled": add `--kernel-trace` before `--stats`
- If RCV window opens on wrong monitor: use `xdotool windowmove <WID> 50 50` to move it
- If RCV shows nothing except Occupancy: the target_cu was not populated by the application. Try a different `att_target_cu` value
- If `INVALID_SHADER_DATA` error: aqlprofile and decoder versions are incompatible. Update both.

## Output

After analysis, provide:
1. The full bottleneck report (as described in Step 6)
2. A prioritized optimization plan with concrete steps
3. Estimated performance improvement for each optimization
4. Suggested follow-up profiling to validate improvements
