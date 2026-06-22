# GLM-4.6 OpenRouter Probe Diagnostics

This run should not yet be interpreted as strong evidence that GLM-4.6 is robust. The paired noisy deltas are non-significant, but the clean baseline appears affected by output-budget behavior.

## Same-Subset Clean Accuracy

| model | correct | total | accuracy |
| --- | ---: | ---: | ---: |
| nano | 191 | 250 | 0.764 |
| haiku | 221 | 250 | 0.884 |
| glm | 169 | 250 | 0.676 |

## Output-Budget Symptoms

- Router max output tokens for this run: 256.
- Clean GLM responses at or above 250 completion tokens: 68 / 250.
- Wrong clean GLM responses at or above 250 completion tokens: 61 / 68.
- Empty clean GLM tool-call predictions: 35 / 250.
- Empty clean GLM predictions at or above 250 completion tokens: 34 / 35.

OpenRouter usage reports reasoning tokens inside completion usage for this model. A 256-token cap can therefore truncate thinking before the final tool call.

## Paired GLM McNemar Results

| dimension | clean acc | noisy acc | drop | clean->noisy fail | noisy->clean fix | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| typos | 0.676 | 0.688 | -0.012 | 12 | 15 | 0.701 |
| cursing | 0.676 | 0.712 | -0.036 | 12 | 21 | 0.163 |
| irrelevant_context | 0.676 | 0.680 | -0.004 | 16 | 17 | 1.000 |
| removed_spaces | 0.676 | 0.692 | -0.016 | 10 | 14 | 0.541 |
| argumentative_challenge | 0.676 | 0.664 | 0.012 | 18 | 15 | 0.728 |
| pasted_context_block | 0.676 | 0.708 | -0.032 | 12 | 20 | 0.215 |
| telegraphic_request | 0.676 | 0.688 | -0.012 | 14 | 17 | 0.720 |

## Diagnostic Conclusion

The observed GLM paired deltas are consistent with no degradation, but the low clean baseline is not clean enough to support a final robustness claim. Rerun a small same-subset GLM diagnostic with `REALISTIC_BFCL_ROUTER_MAX_OUTPUT_TOKENS=1024` before interpreting the model-strength gradient.
