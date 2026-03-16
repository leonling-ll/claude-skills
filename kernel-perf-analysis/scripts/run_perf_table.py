#!/usr/bin/env python3
"""
run_perf_table.py  —  Analysis only.

Parse a rocprofv3 kernel-trace CSV (collected by the att-runner agent) together
with Triton cache .amdgcn metadata and print a markdown performance table.

The att-runner agent is responsible for running rocprofv3 and writing the CSV.
This script only reads and formats the results.

Usage:
    # With a kernel-trace CSV from att-runner:
    python3 run_perf_table.py --csv /path/to/kernel_trace.csv \
        [--kernel-name REGEX] [--iters N] [--tflops FLOAT] \
        [--mfma-eff PCT] [--label LABEL]

    # With a kernel file path to look up Triton cache metadata:
    python3 run_perf_table.py --csv /path/to/kernel_trace.csv \
        --kernel-file my_kernel.py --kernel-name "matmul_kernel" \
        --iters 20 --label "baseline"

Arguments:
    --csv            Path to rocprofv3 *_kernel_trace.csv (required)
    --kernel-file    Original kernel .py file (used to locate Triton .amdgcn metadata)
    --kernel-name    Substring to match kernel rows in the CSV (default: auto-detect)
    --iters          Number of last dispatches to average for timing (default: 20)
    --tflops         TFLOPS value (float) if pre-computed or parsed from stdout
    --mfma-eff       MFMA efficiency string e.g. "57.98%" if already known
    --label          Label for the Version column (default: kernel filename or "kernel")
"""

import argparse
import csv
import glob
import os
import re
import sys


TRITON_CACHE = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def find_kernel_trace_csv(trace_dir):
    """Find *_kernel_trace.csv inside rocprofv3 output (hostname/pid subdirs)."""
    pattern = os.path.join(trace_dir, "*", "*_kernel_trace.csv")
    files = glob.glob(pattern)
    if not files:
        pattern2 = os.path.join(trace_dir, "*_kernel_trace.csv")
        files = glob.glob(pattern2)
    return max(files, key=os.path.getmtime) if files else None


def auto_detect_kernel_name(csv_path):
    """Return the first non-empty Kernel_Name found in the CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Kernel_Name", "").strip()
            if name:
                return name
    return None


def avg_kernel_time_us(csv_path, kernel_name, last_n=20):
    """
    Average elapsed time (µs) for the last `last_n` dispatches matching
    `kernel_name`. Returns (avg_us, total_dispatches).
    """
    durations = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_name in row.get("Kernel_Name", ""):
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                durations.append((end - start) / 1e3)   # ns → µs
    if not durations:
        return None, 0
    tail = durations[-last_n:]
    return sum(tail) / len(tail), len(durations)


def parse_amdgcn_metadata(kernel_name_fragment):
    """
    Parse .vgpr_count and .vgpr_spill_count from the newest matching .amdgcn
    file in the Triton cache.
    """
    pattern = os.path.join(TRITON_CACHE, "*", f"*{kernel_name_fragment}*.amdgcn")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(TRITON_CACHE, "*", "*.amdgcn")
        files = glob.glob(pattern)
    if not files:
        return None, None

    fpath = max(files, key=os.path.getmtime)
    vgprs, spills = None, None
    with open(fpath, "r") as f:
        for line in f:
            vm = re.search(r"\.vgpr_count:\s*(\d+)", line)
            if vm:
                vgprs = int(vm.group(1))
            sm = re.search(r"\.vgpr_spill_count:\s*(\d+)", line)
            if sm:
                spills = int(sm.group(1))
            if vgprs is not None and spills is not None:
                break
    return vgprs, spills


def parse_tflops_from_text(text):
    """
    Parse TFLOPS from free-form text (stdout/stderr).
    Matches patterns like: TFLOPS: 123.4 / 123.4 TFLOPS / 123.4 tflops
    Returns the last match found, or None.
    """
    tflops = None
    for line in text.splitlines():
        m = re.search(r"(\d+\.?\d*)\s*(?:T?FLOPS|tflops)", line, re.IGNORECASE)
        if m:
            tflops = float(m.group(1))
    return tflops


def parse_mfma_efficiency(text):
    """Parse 'mfma efficiency': '57.98%' from process_json.py JSON output."""
    m = re.search(r'"mfma efficiency"\s*:\s*"([\d.]+%)"', text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def fmt(val, spec=None):
    if val is None:
        return "FAIL"
    if spec:
        return spec.format(val)
    return str(val)


def print_table(rows):
    """Print a markdown performance table."""
    print()
    print("| Version              | TFLOPS | VGPRs | Spills | MFMA Eff. | avg time  |")
    print("|----------------------|--------|-------|--------|-----------|-----------|")
    for row in rows:
        label   = row["label"][:20]
        tflops  = fmt(row.get("tflops"),   "{:.0f}")
        vgprs   = fmt(row.get("vgprs"))
        spills  = fmt(row.get("spills"))
        mfma    = fmt(row.get("mfma_eff"))
        avg_us  = row.get("avg_us")
        avgtime = f"{avg_us:.2f} us" if avg_us is not None else "N/A"
        print(f"| {label:<20} | {tflops:>6} | {vgprs:>5} | {spills:>6} | {mfma:>9} | {avgtime:>9} |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse kernel-trace CSV and print a markdown performance table."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to rocprofv3 *_kernel_trace.csv (produced by att-runner agent)",
    )
    parser.add_argument(
        "--kernel-file",
        default="",
        help="Original kernel .py file (used to look up Triton .amdgcn metadata)",
    )
    parser.add_argument(
        "--kernel-name",
        default="",
        help="Kernel name substring for matching CSV rows (default: auto-detect)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of last dispatches to average (default: 20)",
    )
    parser.add_argument(
        "--tflops",
        type=float,
        default=None,
        help="Pre-computed TFLOPS value (optional)",
    )
    parser.add_argument(
        "--mfma-eff",
        default=None,
        help='MFMA efficiency string e.g. "57.98%%" (optional)',
    )
    parser.add_argument(
        "--label",
        default="",
        help="Label for the Version column (default: kernel filename)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    label = args.label
    if not label:
        label = os.path.basename(args.kernel_file) if args.kernel_file else "kernel"

    # Resolve kernel name
    kernel_name = args.kernel_name or auto_detect_kernel_name(csv_path) or ""

    # Timing from CSV
    avg_us, n_dispatches = avg_kernel_time_us(csv_path, kernel_name, last_n=args.iters)
    if avg_us is not None:
        print(f"  {n_dispatches} dispatches found, avg of last {args.iters} = {avg_us:.2f} µs")
    else:
        print(f"  WARNING: no rows matched kernel '{kernel_name}' in {csv_path}")

    # VGPR metadata from Triton cache
    fragment = args.kernel_name or (
        os.path.splitext(os.path.basename(args.kernel_file))[0] if args.kernel_file else ""
    )
    vgprs, spills = parse_amdgcn_metadata(fragment) if fragment else (None, None)

    row = {
        "label":    label,
        "tflops":   args.tflops,
        "vgprs":    vgprs,
        "spills":   spills,
        "mfma_eff": args.mfma_eff,
        "avg_us":   avg_us,
    }
    print_table([row])


if __name__ == "__main__":
    main()
