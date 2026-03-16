---
name: lds-bank-conflict
description: >
  Collect and report LDS (Local Data Share) bank conflict counts for AMD GPU kernels
  using rocprofv3 hardware performance counters. Reports SQ_LDS_BANK_CONFLICT and
  SQ_LDS_DATA_FIFO_FULL per dispatch, per CTA, and provides a severity assessment.
  Use this skill whenever the user asks to measure, check, profile, or count LDS bank
  conflicts in a GPU kernel, or wants to know if a kernel has shared memory bank
  conflicts on AMD GPUs (MI300X, MI308X, MI350, CDNA3/CDNA4).
  Usage: /lds-bank-conflict <run_command> [kernel_regex]
tools: Bash,Read,Glob
---

# LDS Bank Conflict Counter

Measure LDS bank conflicts in AMD GPU kernels using `rocprofv3` hardware counters.

## What This Does

Runs your kernel under `rocprofv3` to collect `SQ_LDS_BANK_CONFLICT` and
`SQ_LDS_DATA_FIFO_FULL` counters, then parses the resulting CSV and prints a
clean per-kernel summary table with a severity interpretation.

## Arguments

- `<run_command>` — the command that launches the kernel (e.g., `python my_kernel.py`)
- `[kernel_regex]` — optional regex to filter to specific kernels (default: all kernels)

## Workflow

### Step 1 — Ensure the counter YAML exists

Look for `lds_bank_conf_counter.yaml` in the current directory. If it is missing,
create it:

```yaml
jobs:
  - pmc:
    - SQ_LDS_BANK_CONFLICT
    - SQ_LDS_DATA_FIFO_FULL
```

### Step 2 — Run rocprofv3

Clean any old output, then collect counters. If the user supplied a kernel regex,
include `--kernel-include-regex`. If not, omit that flag (rocprofv3 captures all kernels).

```sh
OUTPUT_DIR="lds_conflict_counters"
rm -rf "$OUTPUT_DIR"

# With kernel filter:
rocprofv3 -i lds_bank_conf_counter.yaml \
  --kernel-include-regex "<kernel_regex>" \
  -d "$OUTPUT_DIR" -f csv -- <run_command>

# Without kernel filter:
rocprofv3 -i lds_bank_conf_counter.yaml \
  -d "$OUTPUT_DIR" -f csv -- <run_command>
```

### Step 3 — Find and parse the CSV

The output lands in `lds_conflict_counters/pass_1/**/*_counter_collection.csv`.
Use `scripts/parse_lds_conflicts.py` to parse and print the summary:

```sh
python /root/.claude/skills/lds-bank-conflict/scripts/parse_lds_conflicts.py \
    lds_conflict_counters/pass_1/
```

The script will:
- Group rows by kernel name
- Report: kernel name, grid size, workgroup size, number of dispatches captured,
  `SQ_LDS_BANK_CONFLICT` per dispatch, conflicts per CTA
- Interpret severity:
  - `0` → No bank conflicts (ideal)
  - `1–1,000` → Minor
  - `1,001–1,000,000` → Moderate — worth investigating
  - `>1,000,000` → Severe — likely a meaningful performance bottleneck

### Step 4 — Report to user

Present the table produced by the script as-is and add a brief explanation:
- Which kernels have zero conflicts (good)
- Which kernels have conflicts and what the count means

## Notes

- `SQ_LDS_BANK_CONFLICT` is cumulative across all threads in the dispatch; dividing
  by the number of CTAs (= grid_size / workgroup_size) gives conflicts per CTA.
- The counter value is consistent across repeated dispatches of the same kernel in a
  single run, so a single dispatch is representative.
- `LDS_Block_Size = 0` in the dispatch info does **not** mean no LDS is used — the
  compiler may manage LDS internally (e.g., for `tl.dot` in Triton). The hardware
  counter is the ground truth.
- rocprofv3 requires root or the appropriate HSA permissions; if the command fails,
  check that `ROCR_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` is set to a valid GPU.
