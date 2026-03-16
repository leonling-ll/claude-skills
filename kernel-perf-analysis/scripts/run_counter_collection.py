#!/usr/bin/env python3
"""
run_counter_collection.py  —  Analysis only.

Parse a rocprofv3 counter-collection CSV (collected by the att-runner agent)
and print an averaged summary table.

The att-runner agent is responsible for running rocprofv3 with the correct
PMC counter YAML and writing the CSV. This script only reads and formats results.

Usage:
    python3 run_counter_collection.py --csv /path/to/counter_collection.csv \
        --counters SQ_LDS_BANK_CONFLICT,SQ_LDS_DATA_FIFO_FULL \
        [--kernel-name REGEX] [--label LABEL]

Arguments:
    --csv            Path to rocprofv3 *_counter_collection.csv (required)
    --counters       Comma-separated counter names to display (required)
    --kernel-name    Kernel name substring for filtering rows (default: auto-detect)
    --label          Label for the Version column (default: "kernel")
"""

import argparse
import csv
import os
import sys


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def auto_detect_kernel_name(csv_path):
    """Return the first non-empty Kernel_Name found in the counter CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Kernel_Name", "").strip()
            if name:
                return name
    return None


def parse_counter_csv(csv_path, counters, kernel_name):
    """
    Parse counter values from a rocprofv3 counter_collection CSV.

    Returns (averages_dict, n_dispatches) where averages_dict maps
    counter_name -> average float value (None if counter not found).
    n_dispatches is the number of dispatches that matched kernel_name.
    """
    totals = {c: [] for c in counters}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_name and kernel_name not in row.get("Kernel_Name", ""):
                continue
            name = row.get("Counter_Name", "")
            if name in totals:
                try:
                    totals[name].append(float(row["Counter_Value"]))
                except (ValueError, KeyError):
                    pass

    averages = {c: (sum(v) / len(v) if v else None) for c, v in totals.items()}
    n = next((len(v) for v in totals.values() if v), 0)
    return averages, n


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_table(rows, counters):
    """Print a markdown counter summary table."""
    col_w = max(20, max(len(c) for c in counters) + 2)
    header = "| {:<20} | ".format("Version") + " | ".join(
        f"{c:<{col_w}}" for c in counters
    ) + " | Dispatches |"
    sep = (
        "|" + "-" * 22 + "|"
        + "|".join("-" * (col_w + 2) for _ in counters)
        + "|" + "-" * 12 + "|"
    )

    print()
    print(header)
    print(sep)
    for row in rows:
        vals = []
        for c in counters:
            v = row["averages"].get(c)
            vals.append(f"{v:>{col_w},.0f}" if v is not None else f"{'FAIL':>{col_w}}")
        n = row["n_dispatches"]
        disp = f"{n:>10,}" if n else f"{'FAIL':>10}"
        print(f"| {row['label']:<20} | " + " | ".join(vals) + f" | {disp} |")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse counter-collection CSV and print a markdown summary table."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to rocprofv3 *_counter_collection.csv (produced by att-runner agent)",
    )
    parser.add_argument(
        "--counters",
        required=True,
        help="Comma-separated counter names (e.g. SQ_LDS_BANK_CONFLICT,SQ_LDS_DATA_FIFO_FULL)",
    )
    parser.add_argument(
        "--kernel-name",
        default="",
        help="Kernel name substring for filtering rows (default: auto-detect)",
    )
    parser.add_argument(
        "--label",
        default="kernel",
        help="Label for the Version column (default: kernel)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    counters = [c.strip() for c in args.counters.split(",") if c.strip()]
    if not counters:
        print("Error: --counters must be a non-empty comma-separated list")
        sys.exit(1)

    # Resolve kernel name
    kernel_name = args.kernel_name or auto_detect_kernel_name(csv_path) or ""

    averages, n_dispatches = parse_counter_csv(csv_path, counters, kernel_name)
    print(f"  {n_dispatches} dispatches matched kernel '{kernel_name}'")

    row = {"label": args.label, "averages": averages, "n_dispatches": n_dispatches}
    print_table([row], counters)


if __name__ == "__main__":
    main()
