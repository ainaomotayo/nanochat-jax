# Maintainer Communication Templates

Templates for engaging with the NanoChat community and related researchers. Use the decision tree at the bottom before sending anything.

---

## 1. GitHub Issue: Bug Report

Use this if you find an actual bug in the upstream NanoChat repository during the porting process.

```markdown
**Title:** [Bug] <concise description of the bug>

**NanoChat version/commit:** <commit hash>

**Description:**

While porting NanoChat to JAX/Flax NNX ([nanochat-jax](https://github.com/ainaomotayo/nanochat-jax)), I encountered unexpected behavior in `<module/file>`.

**Steps to reproduce:**

1. <step>
2. <step>
3. <step>

**Expected behavior:**

<what should happen>

**Actual behavior:**

<what actually happens>

**Evidence from JAX port:**

During the port, I wrote an equivalent implementation that <describe how the JAX version revealed the bug>. The relevant test case is in `<test file>`.

**Environment:**
- Python: <version>
- PyTorch: <version>
- Hardware: <GPU/TPU>
- OS: <os>

**Suggested fix (optional):**

<If you have a fix or know the root cause, describe it here. Offer to submit a PR if appropriate.>
```

---

## 2. GitHub Discussion: Sharing NanoChat-JAX

This is the most likely template to use. Frame it as contributing to the community, not promoting.

```markdown
**Title:** JAX/Flax NNX port of NanoChat with scaling law experiments

**Category:** Show and Tell

Hi all,

I built a JAX/Flax NNX port of NanoChat and used it to run scaling law experiments across both frameworks. Sharing here in case it's useful to the community.

**What it is:**

- A faithful port of NanoChat to JAX/Flax NNX -- same architecture, same hyperparameters, same training logic
- 180 tests verifying numerical equivalence with the PyTorch implementation
- Scaling law experiments from 886K to 6.03B parameters on TinyShakespeare

**Key findings:**

- The scaling law L = 3.29 x N^(-0.027) holds cleanly on TinyShakespeare, with the shallow exponent consistent with the small dataset size
- JAX is 1.2-2.5x faster on TPU; comparable on single GPU at small scale
- Full write-up: [blog post](docs/research/BLOG_POST.md) and [paper](docs/research/ARXIV_PAPER.md)

**Why JAX:**

The functional paradigm maps well to multi-device training. The port also serves as a cross-framework validation -- having two independent implementations of the same architecture is useful for catching subtle bugs.

**Repo:** https://github.com/ainaomotayo/nanochat-jax

Happy to answer questions or take feedback. If there's interest, I can document the porting process (the tricky parts were weight initialization equivalence and attention mask handling).

Supported by Google TPU Research Cloud and the AI GDE TPU Sprint 2026.
```

---

## 3. Email: Pre-arXiv Review Request

Use only if you have a near-final paper and a specific, concrete ask. Do not send this cold without having first engaged via GitHub Discussion.

```
Subject: Scaling law experiments across JAX and PyTorch using NanoChat -- paper review request

Hi Andrej,

I ported NanoChat to JAX/Flax NNX and ran scaling law experiments across both frameworks (886K to 6.03B parameters on TinyShakespeare). The measured scaling law is L = 3.29 x N^(-0.027), with JAX showing 1.2-2.5x speedups on TPU at larger scales.

The implementation has 180 tests verifying numerical equivalence with the PyTorch version, and the full codebase is public:
https://github.com/ainaomotayo/nanochat-jax

I have a draft paper documenting the methodology and results:
[link to paper draft]

Two specific questions I'd value your input on:

1. <specific technical question about the architecture or methodology>
2. <specific question about how to frame the cross-framework comparison>

I want to make sure the paper accurately represents NanoChat's design decisions before submitting. Happy to incorporate any corrections.

Best,
<name>
```

---

## Decision Tree: When to Engage

Use this before sending any of the above templates.

### Engage if:

- **You found a real bug.** You have a reproducer, you understand the root cause, and your JAX port provides independent evidence. Use Template 1 (GitHub Issue).
- **Your work adds concrete value to the community.** The JAX port is complete, tested, and the scaling law results are verified. You are sharing, not asking for anything. Use Template 2 (GitHub Discussion).
- **You want to cite correctly.** You have a near-final paper and need to verify that your characterization of NanoChat's design decisions is accurate. Use Template 3 (Email), but only after engaging on GitHub first.
- **The JAX port could become an official community resource.** There is community demand for a JAX version, and maintainers have expressed interest.

### Do not engage if:

- **Your framing is "PyTorch is worse."** The findings are about scaling laws and cross-framework validation, not framework ranking. If your messaging reads as adversarial, rewrite it.
- **Benchmarks are unverified.** Do not share performance claims you haven't reproduced at least twice. Do not extrapolate single-run numbers.
- **You are primarily seeking endorsement.** If the main goal is to get Karpathy to retweet or endorse the work, that will come through in the tone. Share the work on its merits and let it stand.
- **You haven't done the public work first.** The GitHub Discussion should come before the email. Build a public record of the contribution before making private requests.

### Engagement order:

1. GitHub Discussion (share the work publicly)
2. GitHub Issue (only if a bug exists)
3. Email (only after public engagement, only with a specific ask)
