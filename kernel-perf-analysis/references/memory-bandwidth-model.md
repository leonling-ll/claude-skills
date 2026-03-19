# Memory Bandwidth Model Reference (AMD GFX9)

This document is a concise reference for interpreting memory-hierarchy profiling
results from `rocprof-compute`. The full derivation is in
`gfx9-gluon-tutorials/docs/memory_bandwidth_model.md`.

---

## Hardware parameters (GFX9)

| Parameter | gfx942 (MI300X/MI308X) | gfx950 (MI350) |
|-----------|------------------------|----------------|
| Peak HBM BW | 5300 GB/s | 8000 GB/s |
| Peak L2 BW (aggregate) | ~7500 GB/s | ~9600 GB/s |
| Peak L1/TCP BW (aggregate) | ~28800 GB/s | ~40960 GB/s |
| L1 (TCP) size per CU | 32 KB | 32 KB |
| Number of CUs | 304 | 256 |
| HBM round-trip latency | ~800 ns | ~800 ns |
| LDS size per CU | 64 KB | 160 KB |

---

## Steady-state bandwidth model

The kernel's achievable HBM bandwidth per CU is:

```
BW_per_CU = effective_inflight_bytes_per_CU / hbm_latency
```

where:

```
effective_inflight_bytes_per_CU =
    min(inflight_bytes_per_CU, CU_inflight_limit)

inflight_bytes_per_CU =
    num_req_per_wave × data_per_request_per_wave × num_active_waves_per_CU

num_req_per_wave =
    min(hbm_latency / effective_compute_latency, num_stages - 1)

effective_compute_latency =
    waves_per_simd × compute_latency

CU_inflight_limit = 32 KB  (TCP size on GFX9)
```

### Key tensions

| Situation | Effect | Fix |
|-----------|--------|-----|
| `compute_latency` >> `hbm_latency / (num_stages-1)` | Wave too slow to fill pipeline | Reduce compute work per iteration, or increase `num_stages` |
| `compute_latency` << `hbm_latency / (num_stages-1)` | Buffer-limited; adds more stages | Increase `num_stages` |
| High occupancy (many `waves_per_simd`) | Slows each wave; reduces per-wave in-flight count | Optimal occupancy < maximum occupancy |
| TCP full (32 KB per CU) | In-flight cap hit; no benefit from more stages | Reduce `data_per_request_per_wave` (smaller tile) |

---

## Little's Law interpretation

Given measured achieved HBM bandwidth `BW_achieved` (GB/s):

```
inflight_bytes ≈ BW_achieved × hbm_latency
               = BW_achieved_Bps × 800e-9
```

Compare to the TCP cap (32 KB × NUM_CUs):
- `inflight_bytes / TCP_total < 50%` → pipeline is under-filled; can add stages
- `inflight_bytes / TCP_total ≈ 100%` → TCP-limited; adding stages won't help

---

## Memory hierarchy roofline thresholds

| Level | Utilization | Interpretation |
|-------|------------|----------------|
| HBM | > 85% | HBM-saturated; kernel is memory-bound at DRAM |
| HBM | 50–85% | Good; small gains possible from pipeline tuning |
| HBM | < 50% | Under-utilizing HBM; request issue rate too low |
| L2 hit rate | > 80% | Excellent reuse; reduces DRAM pressure |
| L2 hit rate | 40–80% | Moderate; check tiling/swizzle |
| L2 hit rate | < 40% | Every access goes to HBM; purely HBM-bound |
| L1 hit rate | > 80% | Working set fits in TCP; compute or LDS bound |
| L1 hit rate | < 50% | TCP thrashing; tile too large for 32 KB per CU |

---

## End-to-end vs. steady-state

`rocprof-compute` reports counters averaged across all dispatches.
For small problem sizes, prologue + epilogue overhead inflates apparent latency:

```
total_cycles = hbm_latency + num_iters × iter_latency + hbm_latency
```

If the kernel runs only a few iterations (`num_iters` small), HBM utilization
will appear low even for a well-tuned pipeline. Use the E2E model:

```
BW_achieved = total_bytes / total_cycles
```

Compare to `BW_per_CU × num_active_CUs` from the steady-state model to assess
whether the gap is pipeline design or problem-size overhead.

---

## Suggested optimization skills

| Bottleneck | Recommended skill |
|------------|-------------------|
| Low HBM utilization, pipeline not full | `/gluon-pipeline-opt` |
| LDS bank conflicts | `/gluon-lds-opt` |
| GPR pressure / loop overhead | `/gluon-gpr-opt` |
| Prologue/epilogue tail latency | `/gluon-beyond-loop-opt` |
| L2 cache reuse (B matrix) | `/gluon-beyond-loop-opt` (XCD remapping) |
