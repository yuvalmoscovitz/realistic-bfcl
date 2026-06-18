# Paper Flow

Realistic-BFCL should read like a compact benchmark paper, not a product report.
The closest structure is shared by robustness benchmarks, behavioral testing
work, and human-audited evaluation projects: define the threat to validity,
control the transformation, run paired evaluation, separate artifacts from true
failures, and show qualitative evidence.

## 1. Motivation

Clean function-calling scores do not prove production robustness. Real users add
casual language, irrelevant context, typos, frustration, and small formatting
mistakes while still making understandable requests.

## 2. Benchmark Construction

BFCL provides the deterministic substrate: tool schemas, prompts, and gold
tool-call oracles. Realistic-BFCL adds oracle-preserving metamorphic prompt
variants on top of the same examples.

The key validity claim is not that every noisy prompt is harder. It is that each
accepted noisy prompt remains a plausible user request with the same final tool
oracle.

## 3. Realism Dimensions

The first version uses five focused dimensions:

- `typos`
- `cursing`
- `irrelevant_context`
- `removed_spaces`
- `argumentative_challenge`

The project deliberately avoids a large perturbation catalog until these
dimensions are audited and stable.

## 4. Evaluation

Evaluation is paired:

- same base BFCL example
- same tool schema
- same model
- clean prompt vs. noisy prompt

The primary signal is `clean_success_noisy_failure`, because it isolates cases
where a model handled the clean BFCL prompt but failed after realistic noise.

## 5. Artifact Separation

The analysis must separate strong model failures from benchmark artifacts:

- `oracle_issue`: accepted aliases or formatting may be too narrow.
- `augmentation_issue`: the noisy prompt may have changed an entity boundary or
  copied corrupted text into the argument.
- `baseline_dataset_issue`: the original BFCL prompt/schema/oracle may be
  ambiguous.
- `real_model_regression`: remaining failures after excluding those artifacts.

This mirrors the best benchmark practice: do not inflate claims by hiding
dataset or evaluator defects inside model-failure counts.

## 6. Qualitative Evidence

The paper should include a small audited table of strong failures:

- missing tool calls in multi-call requests
- wrong function routing
- unnecessary extra tool calls
- clear argument flips

`artifacts/analysis/strong_failure_examples.csv` is the source artifact for this
section.

## 7. Scaling

After the pilot taxonomy is stable, scale by adding more BFCL categories and
then more models. Do not add more augmentations until the current dimensions are
defensible.
