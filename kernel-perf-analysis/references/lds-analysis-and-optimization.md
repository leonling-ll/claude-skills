# LDS Analysis and Optimization Reference

This document distills key concepts of LDS analysis. Use it as a quick
reference when interpreting ATT traces, hardware counter data, or LDS-related
performance anomalies.

---

## 1. `ds_read` Throughput — Mental Model

LDS data flow for a `ds_read` instruction passes through four stages:

1. **Instruction issue (SQ)** — 4 cycles (64 threads / 16-wide SIMD). Not
   a bottleneck; the SQ can over-issue if nothing applies back-pressure.
2. **Address transfer (SIMD → LDS)** — 4 cycles per SIMD pair at 128 B/cycle.
   Fast relative to downstream stages.
3. **LDS service** — LDS has 64 banks × 4 B = 256 B/cycle service bandwidth.
   `ds_read_b128` requests 1024 B per instruction × 4 SIMDs = 4096 B → **16 cycles**.
   This is the first real limit.
4. **Data return (LDS → SIMD)** — 128 B/cycle bus, 2048 B per SIMD pair →
   **16 cycles**. Matches LDS service rate exactly.

**Steady-state throughput rules (bank-conflict-free, 4 waves/CU):**

| Instruction     | Data per instr | Throughput     |
|-----------------|----------------|----------------|
| `ds_read_b128`  | 16 B/thread    | 1 per **16 cycles** |
| `ds_read_b64`   | 8 B/thread     | 1 per **8 cycles**  |

One `ds_read_b128` ≡ two `ds_read_b64` in bandwidth, but `b128` saves
instruction count (better for pipelined kernels).

---

## 2. Diagnosing Bank Conflicts from ATT Traces

**Key rule:** steady-state `ds_read_b128` issue interval directly reveals
bank-conflict severity.

| Observed interval | Diagnosis                  |
|-------------------|----------------------------|
| 16 cycles         | Conflict-free (ideal)      |
| 32 cycles         | 2-way bank conflict        |
| 64 cycles         | 4-way bank conflict        |

### Reading the ATT trace

- Open the `ui_*/` directory produced by rocprofv3 ATT mode in ATTViewer.
- Look at the **thread trace** for `ds_read_b128` instructions.
- Measure the cycle gap between consecutive issue events on the same SIMD.
- If the gap exceeds 16 cycles, bank conflicts are present.

### Hardware counter confirmation

Use `SQ_LDS_BANK_CONFLICT` via rocprofv3 counter mode to get a numeric count:

```yaml
jobs:
  - pmc:
      - SQ_LDS_BANK_CONFLICT
      - SQ_LDS_DATA_FIFO_FULL
```

A non-zero `SQ_LDS_BANK_CONFLICT` count corroborates what the ATT trace shows.

---

## 3. Three LDS Layout Strategies

Bank conflicts are a layout design problem: they arise from the mapping
`(thread_id, wave_id) → LDS bank`. The fix is in the layout.

### 3.1 Raw (no swizzling, no padding)

- **Access pattern:** linear `(row, col) → offset`; simple and easy to reason about.
- **Bank conflicts:** typically 4-way conflicts for standard GEMM tiles
  (e.g., 256×64 operand A with `ds_read_b128`).
- **VGPR usage:** good — constant distance between vectors allows a
  single base VGPR + offsets.
- **Verdict:** register-efficient but severely bank-conflicted.

### 3.2 Swizzling (`SwizzledSharedLayout`)

- **Mechanism:** XORs column bits into row bits to spread accesses across banks.
  Parameters: `(vec, perPhase, maxPhase)` — e.g., `(8, 1, 8)` for conflict-free.
- **Bank conflicts:** eliminated (16-cycle issue interval restored).
- **VGPR usage:** increased — non-constant vector distances require multiple
  base VGPRs, raising register pressure.
- **Side effects:** introduces `ds_bpermute` overhead on the global-load side;
  can trigger different LLVM scheduling heuristics.
- **Verdict:** conflict-free but trades register pressure and code complexity.

### 3.3 Padding (`PaddedSharedLayout`)

- **Mechanism:** inserts padding bytes between rows so that the row stride
  shifts accesses to different banks, without breaking offset linearity.
  Example parameters: `[[512, 16]]` stride with associated basis.
- **Bank conflicts:** eliminated (16-cycle issue interval).
- **VGPR usage:** minimal — constant distance preserved; single base VGPR.
- **LDS overhead:** slightly higher memory usage due to padding bytes.
  Negligible on gfx950 (160 KB LDS/CU), but can constrain tile sizes on
  gfx942 (64 KB LDS/CU).
- **Verdict:** best balance of conflict elimination and register efficiency
  for most workloads.

---

## 4. Block-Level Evaluation Checklist

Do not evaluate a single `ds_read` in isolation. Evaluate the full tile load:

1. **Steady-state issue interval** — 16 cycles (conflict-free) or longer
   (conflicts)? Measure in ATT trace.
2. **Base VGPR count** — how many distinct base VGPRs appear across all
   `ds_read` instructions for the tensor? Fewer is better.
3. **`ds_read` offset encoding** — offsets are 16-bit; max 65535 bytes.
   Very large tiles may force extra base VGPRs regardless of layout.
4. **LLVM scheduling** — swizzling can trigger interleaved MFMA scheduling
   (sometimes beneficial, sometimes not). Padding and raw layouts usually
   emit back-to-back `ds_read` blocks. Treat unexpected scheduling changes
   as a signal to inspect IR.

---

## 5. Latency vs. Throughput — Key Distinction

| Concern    | Root cause               | Solution               |
|------------|--------------------------|------------------------|
| Latency    | Data not ready when used | LDS prefetching (issue `ds_read` early) |
| Throughput | LDS bank conflicts       | Layout fix (swizzle or padding) |

Over-issuing `ds_read` to hide latency only creates back-pressure if
throughput is already saturated. Fix the layout first, then layer prefetching
on top.

**Common mistake:** seeing long `s_waitcnt lgkmcnt(0)` stalls and assuming
the fix is to issue more `ds_read` instructions earlier — this does not help
if the real bottleneck is bank conflicts cutting throughput in half.

---

## 6. Practical Workflow

```
1. Run rocprofv3 ATT mode  →  collect ui_*/ directory
2. Open ATTViewer          →  measure ds_read_b128 issue interval
3. If interval > 16 cy     →  bank conflicts confirmed
4. Run counter mode        →  verify with SQ_LDS_BANK_CONFLICT count
5. Inspect shared layout   →  switch raw → padding (first choice on gfx950)
                               or raw → swizzling (if padding overhead is unacceptable)
6. Re-run ATT              →  confirm interval returns to 16 cy
7. Check VGPR count        →  ensure base VGPR count did not increase unexpectedly
```

---

## 7. Architecture-Specific Notes

| GPU          | LDS per CU | Recommendation                                      |
|--------------|------------|-----------------------------------------------------|
| gfx942 (MI300X / CDNA3) | 64 KB  | Padding may restrict tile size; evaluate swizzling as alternative |
| gfx950 (MI350 / CDNA4)  | 160 KB | Padding overhead negligible; preferred layout strategy |

---

## 8. Source References

- `gfx9-gluon-tutorials/docs/lds_throughput.md` — full derivation of
  `ds_read` steady-state throughput across all pipeline stages.
- `gfx9-gluon-tutorials/kernels/gemm/a16w16/v3_lds/README.md` — case study
  comparing raw, swizzling, and padding layouts with ATT trace evidence.
- [Lei's blog on Triton bespoke layouts](https://www.lei.chat/posts/triton-bespoke-layouts/)
  — mathematical treatment of swizzled and padded shared layouts.
