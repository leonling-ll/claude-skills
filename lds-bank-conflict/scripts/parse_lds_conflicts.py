#!/usr/bin/env python3
"""
Parse rocprofv3 counter collection CSVs under a pass_1/ directory tree and
print an LDS bank conflict summary table.

Usage:
    python parse_lds_conflicts.py <pass_1_dir>
"""

import csv
import sys
import glob
import os
from collections import defaultdict


def find_csvs(pass1_dir):
    pattern = os.path.join(pass1_dir, "**", "*_counter_collection.csv")
    return glob.glob(pattern, recursive=True)


def parse_csvs(csv_files):
    """
    Returns a dict keyed by kernel_name, each value a dict:
      {
        "grid_size": int,
        "workgroup_size": int,
        "dispatches": int,
        "SQ_LDS_BANK_CONFLICT": float | None,
        "SQ_LDS_DATA_FIFO_FULL": float | None,
      }
    """
    kernels = defaultdict(lambda: {
        "grid_size": None,
        "workgroup_size": None,
        "dispatches": 0,
        "SQ_LDS_BANK_CONFLICT": None,
        "SQ_LDS_DATA_FIFO_FULL": None,
    })

    for csv_file in csv_files:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["Kernel_Name"].strip().strip('"')
                counter = row["Counter_Name"].strip().strip('"')
                value = float(row["Counter_Value"])
                grid = int(row["Grid_Size"])
                wg = int(row["Workgroup_Size"])

                k = kernels[name]
                k["grid_size"] = grid
                k["workgroup_size"] = wg

                if counter in ("SQ_LDS_BANK_CONFLICT", "SQ_LDS_DATA_FIFO_FULL"):
                    if k[counter] is None:
                        k[counter] = value
                    # Track dispatch count by counting SQ_LDS_BANK_CONFLICT rows
                    if counter == "SQ_LDS_BANK_CONFLICT":
                        k["dispatches"] += 1

    return kernels


def severity(conflicts_per_cta):
    if conflicts_per_cta == 0:
        return "None (ideal)"
    elif conflicts_per_cta <= 1_000:
        return "Minor"
    elif conflicts_per_cta <= 1_000_000:
        return "Moderate"
    else:
        return "Severe"


def print_table(kernels):
    if not kernels:
        print("No counter data found.")
        return

    # Column widths
    name_w = max(len(n) for n in kernels) + 2
    name_w = max(name_w, len("Kernel"))

    header = (
        f"{'Kernel':<{name_w}}  {'Grid':>10}  {'WG':>6}  {'Dispatches':>10}  "
        f"{'LDS_BANK_CONFLICT/dispatch':>26}  {'Conflicts/CTA':>14}  {'Severity':<20}  "
        f"{'DATA_FIFO_FULL/dispatch':>23}"
    )
    sep = "-" * len(header)

    print()
    print("=== LDS Bank Conflict Summary ===")
    print(sep)
    print(header)
    print(sep)

    for name, k in sorted(kernels.items()):
        grid = k["grid_size"] or 0
        wg = k["workgroup_size"] or 1
        dispatches = k["dispatches"]
        bc = k["SQ_LDS_BANK_CONFLICT"]
        fifo = k["SQ_LDS_DATA_FIFO_FULL"]

        num_ctas = grid // wg if wg else 0
        conflicts_per_cta = int(bc) // num_ctas if (bc is not None and num_ctas > 0) else 0

        bc_str = f"{int(bc):,}" if bc is not None else "N/A"
        fifo_str = f"{int(fifo):,}" if fifo is not None else "N/A"
        cta_str = f"{conflicts_per_cta:,}"
        sev = severity(conflicts_per_cta) if bc is not None else "N/A"

        print(
            f"{name:<{name_w}}  {grid:>10,}  {wg:>6,}  {dispatches:>10,}  "
            f"{bc_str:>26}  {cta_str:>14}  {sev:<20}  {fifo_str:>23}"
        )

    print(sep)
    print()
    print("Severity scale (conflicts per CTA):")
    print("  0              → None (ideal)")
    print("  1 – 1,000      → Minor")
    print("  1,001 – 1,000,000 → Moderate — worth investigating")
    print("  > 1,000,000    → Severe — likely a meaningful perf bottleneck")
    print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pass_1_dir>")
        sys.exit(1)

    pass1_dir = sys.argv[1]
    csv_files = find_csvs(pass1_dir)

    if not csv_files:
        print(f"No *_counter_collection.csv files found under: {pass1_dir}")
        sys.exit(1)

    kernels = parse_csvs(csv_files)
    print_table(kernels)


if __name__ == "__main__":
    main()
