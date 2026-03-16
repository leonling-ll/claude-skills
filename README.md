# Claude Skills

A collection of Claude Code skills for AMD GPU kernel optimization.

## What are Claude Skills?

Claude Code skills are reusable, slash-command-invokable prompts that encode
expert knowledge into actionable workflows. Each skill lives in its own
directory with a `SKILL.md` file that defines the command name, description,
and step-by-step instructions.

To use a skill, type `/skill-name` in Claude Code. The skill is loaded as
context and Claude follows its workflow.

## Skills in This Repository

### GEMM Kernel Optimization Series

A progressive optimization tutorial for `fp16 × fp16 → fp16` GEMM on AMD CDNA3/4 GPUs
(MI300X / MI308X / MI350) using the [Gluon DSL](https://github.com/ROCm/triton).
Based on the a16w16 tutorial at `/home/leling/gfx9-gluon-tutorials/kernels/gemm/a16w16/`.

| Skill | Optimization | GPU Requirement |
|-------|-------------|-----------------|
| [`/gluon-mm-inst-opt`](gluon-mm-inst-opt/SKILL.md) | **Step A**: Replace `gl.load/store` with `buffer_load/store` (all CDNA); **Step B**: DMA from global to LDS via `async_copy` (CDNA4 only) | CDNA3 + CDNA4 |
| [`/gluon-lds-opt`](gluon-lds-opt/SKILL.md) | Fix LDS bank conflicts via swizzling or padding layout | CDNA3 + CDNA4 |
| [`/gluon-pipeline-opt`](gluon-pipeline-opt/SKILL.md) | Global prefetch (double-buffering) + local prefetch to hide vmcnt/lgkmcnt stalls | CDNA3 + CDNA4 |
| [`/gluon-gpr-opt`](gluon-gpr-opt/SKILL.md) | Loop unroll ×2 with static buffer indices + N-slice B tile to reduce GPR pressure | **CDNA4 only** |
| [`/gluon-beyond-loop-opt`](gluon-beyond-loop-opt/SKILL.md) | XCD-aware PID remapping + epilogue M-slicing interleaved with stores | CDNA3 + CDNA4 |

### GPU Kernel Profiling and Optimization

| Skill | Purpose |
|-------|---------|
| [`/kernel-perf-analysis`](kernel-perf-analysis/SKILL.md) | Collect AMD GPU kernel performance metrics (TFLOPS, VGPRs, MFMA efficiency, hardware counters, ATT traces) using `rocprofv3` |
| [`/lds-bank-conflict`](lds-bank-conflict/SKILL.md) | Measure LDS bank conflicts (`SQ_LDS_BANK_CONFLICT`) per dispatch and CTA using `rocprofv3` hardware counters |

### Kernel Translation

| Skill | Purpose |
|-------|---------|
| [`/triton-to-gluon`](triton-to-gluon/SKILL.md) | Translate a `@triton.jit` kernel to a `@gluon.jit` kernel with explicit AMD layouts (BlockedLayout, AMDMFMALayout, SwizzledSharedLayout) |

## Installation

Skills in this repo are loaded by creating symlinks under `~/.claude/skills/`:

```bash
git clone https://github.com/leonling-ll/claude-skills.git ~/claude-skills
cd ~/claude-skills
for skill in */; do
    skill="${skill%/}"
    ln -sf "$PWD/$skill" ~/.claude/skills/$skill
done
```

## Hardware Compatibility

| GPU | Architecture | Instruction Set | async_copy |
|-----|-------------|-----------------|-----------|
| MI300X | gfx942 (CDNA3) | cdna3 | No |
| MI308X | gfx942 (CDNA3) | cdna3 | No |
| MI350  | gfx950 (CDNA4) | cdna3 + cdna4 | Yes |

Skills that require `gl.amd.cdna4.async_copy` (Step B of `gluon-mm-inst-opt`,
and `gluon-gpr-opt`) are only applicable on MI350 (gfx950). All other skills
apply to both CDNA3 and CDNA4.
