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
| [`/gemm-v3-lds-layout`](gemm-v3-lds-layout/SKILL.md) | Fix LDS bank conflicts via swizzling or padding layout | CDNA3 + CDNA4 |
| [`/gemm-v4-global-prefetch`](gemm-v4-global-prefetch/SKILL.md) | Double-buffer DMA + `wait_group(1)` to hide global memory latency | **CDNA4 only** |
| [`/gemm-v5-local-prefetch`](gemm-v5-local-prefetch/SKILL.md) | Pre-load LDS data into registers one iteration early, hiding `lgkmcnt` stalls | **CDNA4 only** |
| [`/gemm-v6-loop-unroll`](gemm-v6-loop-unroll/SKILL.md) | Unroll main loop ×2 with static buffer indices, eliminate `k%2` modulo | **CDNA4 only** |
| [`/gemm-v7-n-slice`](gemm-v7-n-slice/SKILL.md) | Split B tile into left/right halves for finer-grained async copy groups | **CDNA4 only** |
| [`/gemm-v8-beyond-hotloop`](gemm-v8-beyond-hotloop/SKILL.md) | XCD-aware PID remapping + epilogue M-slicing interleaved with stores | CDNA3 + CDNA4 |

### GPU Kernel Profiling and Optimization

| Skill | Purpose |
|-------|---------|
| [`/kernel-trace-analysis`](kernel-trace-analysis/SKILL.md) | Profile with `rocprofv3` ATT traces, identify bottlenecks (barrier stalls, idle, TA-blocked loads) |
| [`/lds-optimization`](lds-optimization/SKILL.md) | Diagnose and fix LDS bank conflicts and `lgkmcnt` stalls via swizzle/padding |
| [`/prefetch-data-load`](prefetch-data-load/SKILL.md) | Apply software prefetch (double-buffering) to Triton/Gluon/FlyDSL kernel loops |

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

Skills that require `gl.amd.cdna4.async_copy` (v2–v7) are only applicable
on MI350 (gfx950). v1, v3, and v8 apply to both CDNA3 and CDNA4.
