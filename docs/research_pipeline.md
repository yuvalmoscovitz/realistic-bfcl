# Research Pipeline

This repository is organized around a staged research pipeline. Each stage has
an explicit purpose, expected inputs, and expected outputs. Later engineering
can replace the placeholders without changing the benchmark contract.

## Stage 1: Freeze BFCL

Purpose: make the clean substrate reproducible.

Inputs:
- BFCL upstream dataset and evaluator.
- Chosen clean subset definition.

Outputs:
- Pinned dataset commit.
- Pinned evaluator version.
- Pinned model list.
- Frozen clean subset manifest.

Exit criteria:
- Another researcher can reconstruct the same clean examples and tool schemas.

## Stage 2: Clean Baseline

Purpose: verify that this repository reproduces BFCL-style clean scores before
adding any noise.

Inputs:
- Frozen BFCL subset.
- BFCL evaluator adapter.
- Model list.

Outputs:
- Clean per-example predictions.
- Clean aggregate accuracy.
- Setup notes for any deviation from expected BFCL behavior.

Exit criteria:
- Clean scores are understood well enough that later failures can be attributed
  to realistic transformations rather than evaluator wiring.

## Stage 3: Realism Contract

Purpose: define what makes a noisy example valid.

Rules:
- Preserve the BFCL gold function name and arguments.
- Do not add accidental constraints.
- Do not remove required constraints.
- Keep prompts human-plausible and production-like.
- Reject random adversarial perturbations.
- For correction or self-repair, derive a final oracle from the clean oracle.

Outputs:
- Rejection criteria.
- Audit checklist.
- Automatic invariant checks where possible.

## Stage 4: Augmentation Engine

Purpose: generate a small number of high-realism variants before scaling.

Initial dimensions:
- Conversational overhang.
- Incremental slot revelation.

Outputs:
- Candidate noisy examples.
- Transformation metadata.
- Links back to the base BFCL example.

Exit criteria:
- Each candidate has an explicit transformation dimension and oracle-preserving
  rationale.

## Stage 5: Verification

Purpose: prevent invalid noisy examples from entering evaluation.

Checks:
- Tool schema unchanged.
- Function name unchanged unless a repair rule applies.
- Required arguments unchanged unless a repair rule applies.
- No new constraints are introduced.
- No required constraints are removed.
- Prompt remains plausible for a real user.

Outputs:
- Accepted noisy examples.
- Rejected examples with reasons.
- Audit log.

## Stage 6: Paired Evaluation

Purpose: compare clean and noisy performance under identical schemas and models.

Outputs:
- Clean predictions.
- Noisy predictions.
- Per-example paired outcomes.
- Conditional failures where clean succeeds and noisy fails.

## Stage 7: Analysis

Metrics:
- Clean accuracy.
- Noisy accuracy.
- Degradation ratio.
- Conditional failure rate given clean success.

Error taxonomy:
- Routing error.
- Argument drop.
- Argument hallucination.
- Malformed call.
- False refusal.
- Unnecessary clarification question.

## Stage 8: Defense Ablations

Candidate defenses:
- Denoising prompt layer.
- Stricter tool-use instruction.
- Schema formatting variants.
- Structured decoding where available.

Outputs:
- Defense-specific paired scores.
- Residual failure taxonomy.

## Stage 9: Paper And Release

Release artifacts:
- Dataset generation code.
- Accepted noisy examples.
- Evaluator adapter.
- Metrics scripts.
- Analysis scripts.
- Documentation sufficient for reproduction.
