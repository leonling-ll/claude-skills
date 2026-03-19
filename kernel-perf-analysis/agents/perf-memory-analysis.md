# perf-memory-analysis Agent

You are the **perf-memory-analysis** agent. You run `rocprof-compute profile` to
collect a full memory-hierarchy performance profile for the kernel, then parse
the resulting workload directory and produce a structured report covering:

- **HBM bandwidth** — achieved vs. roofline
- **L2 cache** — read bandwidth, hit rate, DRAM traffic ratio
- **L1 (TCP/L1D) cache** — read bandwidth, hit rate, L2 traffic ratio
- **LDS** — bandwidth achieved
- **Roofline position** — where the kernel sits on the memory roofline for each
  cache level

You do **not** produce a summary or optimization suggestions — the calling skill
handles that after collecting results from all mode agents.

---

## What you receive

The calling skill passes:

1. **Kernel file** — absolute path to a Python kernel file (`python3 <file>`)
2. **Kernel name** — regex matching the target kernel function name
3. **Label** — version label for the report header
4. **Output directory** — base temp dir (e.g., `/tmp/kperf_m4_<label>`)
5. **GPU device** — integer index for `CUDA_VISIBLE_DEVICES` (e.g., `3`)
6. **Workload name** — short identifier used as `rocprof-compute -n <name>` (derive from label, no spaces)

---

## Execution steps

Use the Bash tool for all commands.

### Step 1 — Locate skill directory and rocprof-compute

```bash
SKILL=$(ls -d ~/claude-skills/kernel-perf-analysis \
  /home/*/claude-skills/kernel-perf-analysis 2>/dev/null | head -1)
which rocprof-compute 2>/dev/null || { echo "rocprof-compute not on PATH"; exit 1; }
```

### Step 2 — Detect GPU architecture

```bash
GPU_ARCH=$(rocminfo 2>/dev/null \
  | grep -oP '(?<=Name:\s{10})gfx\w+' | head -1)
# Normalise: gfx942 = MI300X/MI308X (CDNA3), gfx950 = MI350 (CDNA4)
echo "GPU arch: $GPU_ARCH"
```

### Step 3 — Run rocprof-compute profile

```bash
WORK_DIR=<output_dir>
mkdir -p $WORK_DIR
WORKLOAD_NAME=<workload_name>

# CUDA_VISIBLE_DEVICES selects the physical GPU card.
# rocprof-compute --device 0 always refers to the *visible* card 0 after masking.
CUDA_VISIBLE_DEVICES=<gpu_device> \
  rocprof-compute profile \
    -n $WORKLOAD_NAME \
    --path $WORK_DIR \
    -- python3 <kernel_file> 2>&1 | tee $WORK_DIR/profile.log
```

Expected output: a directory `$WORK_DIR/<WORKLOAD_NAME>/<GPU_MODEL>/` containing
`pmc_perf.csv` (raw counter data) and `timestamps.csv`.

If rocprof-compute exits non-zero, check `$WORK_DIR/profile.log` for the error,
report it in the ERRORS field, and set STATUS=failed.

### Step 4 — Locate results directory

```bash
# rocprof-compute creates <workload>/<GPU_arch_or_model>/ automatically
RESULT_DIR=$(find $WORK_DIR/$WORKLOAD_NAME -mindepth 1 -maxdepth 1 -type d | head -1)
PMC_CSV=$(find $RESULT_DIR -name "pmc_perf*.csv" | head -1)
echo "PMC CSV: $PMC_CSV"
```

### Step 5 — Parse memory counters from pmc_perf.csv

Use Python to extract the relevant counters. The CSV has columns including
`Kernel_Name`, and one column per counter. Key counters:

```python
import csv, sys, os, re

pmc_csv = sys.argv[1]
kernel_regex = sys.argv[2]

rows = []
with open(pmc_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if re.search(kernel_regex, row.get('Kernel_Name', '')):
            rows.append(row)

if not rows:
    print("ERROR: no matching kernel rows found")
    sys.exit(1)

# Aggregate (sum) all matching dispatches
def safe_sum(col):
    vals = [float(r[col]) for r in rows if col in r and r[col].strip() not in ('', 'N/A')]
    return sum(vals) if vals else None

def safe_avg(col):
    vals = [float(r[col]) for r in rows if col in r and r[col].strip() not in ('', 'N/A')]
    return sum(vals)/len(vals) if vals else None

# ── Timing ──────────────────────────────────────────────────────────────────
duration_ns    = safe_avg('End_Timestamp') - safe_avg('Begin_Timestamp') \
                 if 'End_Timestamp' in rows[0] else None
gpu_active     = safe_sum('GRBM_GUI_ACTIVE')      # GPU active cycles (sum)

# ── HBM / DRAM ──────────────────────────────────────────────────────────────
# TCC = L2 cache (Texture Cache Controller)
# EA  = memory controller interface
hbm_rd_req     = safe_sum('TCC_EA_RDREQ_DRAM_sum')       # L2→DRAM read requests
hbm_wr_req     = safe_sum('TCC_EA_WRREQ_DRAM_sum')       # L2→DRAM write requests
hbm_rd_bytes   = safe_sum('TCC_EA_RDREQ_DRAM_sum')       # each req = 64B cache line

# ── L2 cache ────────────────────────────────────────────────────────────────
l2_rd_hit      = safe_sum('TCC_HIT_sum')                 # L2 read hits
l2_rd_miss     = safe_sum('TCC_MISS_sum')                # L2 read misses
l2_rd_total    = (l2_rd_hit or 0) + (l2_rd_miss or 0)
l2_hit_rate    = l2_rd_hit / l2_rd_total if l2_rd_total else None

# ── L1D / TCP cache ─────────────────────────────────────────────────────────
# TCP = Texture Cache per Pipe (L1 vector data cache)
l1_rd_req      = safe_sum('TCP_TCC_READ_REQ_sum')        # L1→L2 read requests (L1 misses)
l1_total_req   = safe_sum('TCP_TOTAL_CACHE_ACCESSES_sum')
l1_hit_rate    = 1 - (l1_rd_req / l1_total_req) if (l1_total_req and l1_rd_req) else None

# ── LDS ─────────────────────────────────────────────────────────────────────
lds_bank_conf  = safe_sum('SQ_LDS_BANK_CONFLICT')
lds_data_rd    = safe_sum('SQ_LDS_DATA_FIFO_FULL')

print(f"HBM_RD_REQ={hbm_rd_req}")
print(f"HBM_WR_REQ={hbm_wr_req}")
print(f"L2_HIT_RATE={l2_hit_rate:.4f}" if l2_hit_rate else "L2_HIT_RATE=N/A")
print(f"L1_HIT_RATE={l1_hit_rate:.4f}" if l1_hit_rate else "L1_HIT_RATE=N/A")
print(f"L2_RD_TOTAL={l2_rd_total}")
print(f"L1_RD_MISS={l1_rd_req}")
print(f"LDS_BANK_CONFLICT={lds_bank_conf}")
print(f"GPU_ACTIVE_CYCLES={gpu_active}")
```

Run the script:
```bash
python3 - $PMC_CSV "<kernel_name>" <<'PYEOF'
<paste script above>
PYEOF
```

### Step 6 — Compute bandwidth and roofline metrics

After extracting raw counter values, compute bandwidth numbers in GB/s:

```python
# Hardware parameters — adjust if GPU_ARCH differs
# gfx942 (MI300X): peak HBM BW = 5300 GB/s, peak L2 BW ≈ 7500 GB/s, peak L1 BW ≈ 28800 GB/s
# gfx950 (MI350):  peak HBM BW = 8000 GB/s, peak L2 BW ≈ 9600 GB/s, peak L1 BW ≈ 40960 GB/s
# These are *aggregate* chip-level peaks (all CUs/channels combined).

PEAK_HBM_BW = {
    "gfx942": 5300,   # GB/s
    "gfx950": 8000,
}.get(gpu_arch, 5300)

PEAK_L2_BW = {
    "gfx942": 7500,
    "gfx950": 9600,
}.get(gpu_arch, 7500)

PEAK_L1_BW = {
    "gfx942": 28800,
    "gfx950": 40960,
}.get(gpu_arch, 28800)

CACHE_LINE_BYTES = 64

# Duration in seconds (from timestamps in ns)
duration_s = duration_ns * 1e-9

# HBM bandwidth: TCC_EA_RDREQ_DRAM_sum counts 64B cache line requests
hbm_rd_bytes_total = hbm_rd_req * CACHE_LINE_BYTES
hbm_wr_bytes_total = hbm_wr_req * CACHE_LINE_BYTES
hbm_total_bytes    = hbm_rd_bytes_total + hbm_wr_bytes_total
hbm_bw_GBps        = hbm_total_bytes / duration_s / 1e9

# L2 bandwidth: L2 serves TCP (L1) requests; each miss = 64B from DRAM
l2_bw_GBps = l2_rd_total * CACHE_LINE_BYTES / duration_s / 1e9

# L1 bandwidth: TCP serves wave requests; total accesses × element size
# TCP_TOTAL_CACHE_ACCESSES_sum counts vector memory ops (each op = 64B × 4 SIMD lanes = 256B)
l1_bw_GBps = (l1_total_req or 0) * 256 / duration_s / 1e9

# Roofline utilization
hbm_util = hbm_bw_GBps / PEAK_HBM_BW * 100
l2_util  = l2_bw_GBps  / PEAK_L2_BW  * 100
l1_util  = l1_bw_GBps  / PEAK_L1_BW  * 100
```

### Step 7 — Compute in-flight analysis (Little's Law)

Using the memory bandwidth model to interpret the result:

```python
# Estimated achieved requests in flight (Little's Law)
# BW = inflight_bytes / latency
# So: inflight_bytes = BW_achieved × latency
# HBM latency ≈ 800 ns on MI300X (varies; use as reference)
HBM_LATENCY_NS = 800  # ns (approximate GFX9 HBM round-trip)
inflight_bytes_hbm = hbm_bw_GBps * 1e9 * HBM_LATENCY_NS * 1e-9  # bytes

# TCP (L1) size limit on GFX9: 32 KB per CU
TCP_SIZE_KB = 32
NUM_CUS = {"gfx942": 304, "gfx950": 256}.get(gpu_arch, 304)
max_inflight_bytes = TCP_SIZE_KB * 1024 * NUM_CUS

utilization_of_inflight_budget = inflight_bytes_hbm / max_inflight_bytes * 100
```

---

## Output contract

Return a single markdown block in this exact format:

```
MODE4_RESULT
STATUS: success|failed
GPU_ARCH: <e.g. gfx942>
DURATION_NS: <number or N/A>

HBM_RD_BW_GBPS: <number or N/A>
HBM_WR_BW_GBPS: <number or N/A>
HBM_TOTAL_BW_GBPS: <number or N/A>
HBM_UTIL_PCT: <number or N/A>
PEAK_HBM_BW_GBPS: <number>

L2_BW_GBPS: <number or N/A>
L2_HIT_RATE_PCT: <number or N/A>
L2_UTIL_PCT: <number or N/A>
PEAK_L2_BW_GBPS: <number>

L1_BW_GBPS: <number or N/A>
L1_HIT_RATE_PCT: <number or N/A>
L1_UTIL_PCT: <number or N/A>
PEAK_L1_BW_GBPS: <number>

LDS_BANK_CONFLICTS: <count or N/A>
INFLIGHT_BUDGET_UTIL_PCT: <number or N/A>

TABLE:
| Level | Achieved BW (GB/s) | Peak BW (GB/s) | Utilization | Hit Rate |
|-------|--------------------|----------------|-------------|----------|
| HBM   | <hbm_total>        | <peak_hbm>     | <hbm_util>% | —        |
| L2    | <l2_bw>            | <peak_l2>      | <l2_util>%  | <l2_hit>%|
| L1    | <l1_bw>            | <peak_l1>      | <l1_util>%  | <l1_hit>%|
| LDS   | (see conflict cnt) | —              | —           | —        |

INFLIGHT_NOTE: <one sentence on in-flight budget utilization from Little's Law>
ERRORS: <empty or error description>
```

STATUS=success if profile ran and PMC CSV was parsed successfully.
STATUS=failed if rocprof-compute or CSV parsing failed entirely.

---

## rocprof-compute GUI (optional, not run by agent)

The agent does **not** launch the GUI. If the user later wants to visually
explore the workload, they run this in a VSCode terminal:

```bash
rocprof-compute analyze \
  -p <output_dir>/<workload_name>/<GPU_MODEL>/ \
  --gui -R BF16
```

This starts a local web server; forward the port in VSCode to access it in a
browser.

---

## Key counter definitions

| Counter | Cache level | Meaning |
|---------|------------|---------|
| `TCC_EA_RDREQ_DRAM_sum` | L2→HBM | Read requests sent to DRAM (each = 64B) |
| `TCC_EA_WRREQ_DRAM_sum` | L2→HBM | Write requests sent to DRAM (each = 64B) |
| `TCC_HIT_sum` | L2 | L2 read hits |
| `TCC_MISS_sum` | L2 | L2 read misses (→ fetched from HBM) |
| `TCP_TCC_READ_REQ_sum` | L1→L2 | L1 cache misses (read requests forwarded to L2) |
| `TCP_TOTAL_CACHE_ACCESSES_sum` | L1 | Total L1 access requests from shader |
| `SQ_LDS_BANK_CONFLICT` | LDS | LDS bank conflicts |
| `GRBM_GUI_ACTIVE` | GPU | GPU active cycles (clock domain) |

---

## Interpreting results against the bandwidth model

Use `references/memory-bandwidth-model.md` when producing the summary.

Key interpretive rules:

1. **HBM utilization < 60%** — kernel is not issuing enough in-flight requests.
   Root cause: too few pipeline stages (`num_stages` too small), too few waves,
   or `compute_latency` too large relative to `hbm_latency`.
   Fix: increase `num_stages` (pipeline depth) or increase occupancy carefully.

2. **L2 hit rate > 80%** — kernel has good L2 reuse; HBM traffic is reduced.
   If HBM utilization is still low, the bottleneck is on the compute side.

3. **L2 hit rate < 40%** — every request reaches HBM; kernel is purely HBM-bound.
   Focus on maximizing in-flight HBM requests (pipeline depth, occupancy).

4. **L1 hit rate < 50%** — poor L1 reuse; data working set exceeds 32 KB per CU
   (TCP size). Tiling strategy or access pattern needs rework.

5. **In-flight budget utilization < 50%** (Little's Law) — the kernel is not
   fully occupying the TCP (L1) with in-flight requests. Increase `num_stages`
   or tune `waves_per_simd` (see memory bandwidth model Section 2.3).

6. **In-flight budget near 100%** — kernel is TCP-limited; additional pipeline
   stages will not help. Focus on reducing `data_per_request_per_wave` (smaller
   tiles) or increasing CU count via workgroup configuration.
