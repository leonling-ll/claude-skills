#!/usr/bin/env python3
"""
run_att.py  —  Analysis only.

Post-process an ATT trace directory (collected by the att-runner agent) using
process_json.py and print MFMA efficiency, loop durations, and per-iteration
timing.

The att-runner agent is responsible for running rocprofv3 --att and producing
the ui_* output directory. This script only reads and analyzes that output.

Usage:
    python3 run_att.py --ui-dir /path/to/ui_matmul_kernel

Arguments:
    --ui-dir    Path to the ui_* directory produced by rocprofv3 ATT (required)
"""

import argparse
import json
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_process_json(ui_dir):
    """
    Run process_json.py on the ui_* ATT output directory.
    Returns the parsed JSON dict, or None on failure.
    """
    process_json = os.path.join(SCRIPT_DIR, "process_json.py")
    if not os.path.isfile(process_json):
        raise FileNotFoundError(
            f"process_json.py not found at {process_json}. "
            "Make sure it is in the scripts/ directory."
        )

    result = subprocess.run(
        [sys.executable, process_json, ui_dir],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: process_json.py failed (exit code {result.returncode})")
        if result.stderr:
            print(result.stderr[-2000:])
        return None

    # Print raw output for inspection
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Try to parse and return structured result
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def summarize(data):
    """Print a human-readable summary of the ATT analysis."""
    if data is None:
        return
    mfma_eff = data.get("mfma efficiency", "N/A")
    avg_iter = data.get("average_iteration_duration")
    loop_ratio = data.get("loop_ratio", "N/A")
    pro_ratio = data.get("pro_ratio", "N/A")
    epi_ratio = data.get("epi_ratio", "N/A")
    num_iter = data.get("num_iterations", "N/A")

    print("\n--- ATT Analysis Summary ---")
    print(f"  MFMA efficiency      : {mfma_eff}")
    print(f"  Avg iteration cycles : {avg_iter:.1f}" if avg_iter else "  Avg iteration cycles : N/A")
    print(f"  Loop iterations      : {num_iter}")
    print(f"  Time distribution    : prologue={pro_ratio}, loop={loop_ratio}, epilogue={epi_ratio}")

    if mfma_eff and mfma_eff != "N/A":
        try:
            pct = float(mfma_eff.rstrip("%"))
            if pct < 60:
                print("\n  ⚠ MFMA efficiency < 60%: kernel is memory-bound.")
                print("    Consider /gluon-pipeline-opt (prefetch) or /gluon-lds-opt (bank conflicts).")
            elif pct < 80:
                print("\n  ⚠ MFMA efficiency 60-80%: room for improvement.")
            else:
                print("\n  ✓ MFMA efficiency > 80%: kernel is compute-bound (good).")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze ATT trace output from att-runner agent."
    )
    parser.add_argument(
        "--ui-dir",
        required=True,
        help="Path to the ui_* directory produced by rocprofv3 ATT (from att-runner agent)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ui_dir = os.path.abspath(args.ui_dir)
    if not os.path.isdir(ui_dir):
        print(f"Error: ui_dir not found: {ui_dir}")
        sys.exit(1)

    print(f"Analyzing ATT trace: {ui_dir}")
    data = run_process_json(ui_dir)
    summarize(data)


if __name__ == "__main__":
    main()
