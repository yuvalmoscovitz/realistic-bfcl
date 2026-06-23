# GLM-4.6 OpenRouter Diagnostics

The first GLM-4.6 OpenRouter run used the default 256-token router output cap. That run should not be interpreted as a clean robustness result: the clean baseline was artificially depressed by completion-budget truncation.

## Same-Subset Clean Accuracy

| run | correct | total | accuracy | empty tool-call predictions | cap reference | responses at/above cap reference | wrong at/above cap reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| glm_4_6_openrouter_deepinfra_cap256 | 169 | 250 | 0.676 | 35 | 250 | 68 | 61 |
| glm_4_6_openrouter_deepinfra_cap1024 | 217 | 250 | 0.868 | 1 | 1000 | 0 | 0 |
| gpt_5_4_nano_same250 | 191 | 250 | 0.764 | 0 | 250 |  |  |
| claude_haiku_4_5_same250 | 221 | 250 | 0.884 | 0 | 250 |  |  |

## Cap Effect

| transition from 256-cap run to 1024-cap run | count |
| --- | ---: |
| wrong_to_correct | 51 |
| correct_to_wrong | 3 |
| both_correct | 166 |
| both_wrong | 30 |

## Interpretation

Raising the router output cap from 256 to 1024 changed GLM clean accuracy from 0.676 to 0.868 on the same 250 examples. Empty tool-call predictions fell from 35 to 1, and 51 examples moved from wrong to correct. This confirms the earlier 256-token GLM paired run is not suitable for model-strength conclusions.

The 1024-cap clean baseline is now comparable to Haiku on the same 250 examples. To evaluate whether GLM is robust to the seven noisy dimensions, the noisy paired GLM run should be repeated with `REALISTIC_BFCL_ROUTER_MAX_OUTPUT_TOKENS=1024`.

## Cap-1024 Paired Result

The seven article-facing noisy dimensions were rerun on the same 250-example pool with GLM-4.6 through OpenRouter, pinned to DeepInfra, at temperature 0 and `REALISTIC_BFCL_ROUTER_MAX_OUTPUT_TOKENS=1024`.

| dimension | clean acc | noisy acc | drop | clean-to-noisy failures | noisy-to-clean fixes | McNemar exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| typos | 0.868 | 0.864 | 0.004 | 7 | 6 | 1.000 |
| cursing | 0.868 | 0.852 | 0.016 | 10 | 6 | 0.454 |
| irrelevant_context | 0.868 | 0.864 | 0.004 | 8 | 7 | 1.000 |
| removed_spaces | 0.868 | 0.848 | 0.020 | 7 | 2 | 0.180 |
| argumentative_challenge | 0.868 | 0.868 | 0.000 | 6 | 6 | 1.000 |
| pasted_context_block | 0.868 | 0.860 | 0.008 | 11 | 9 | 0.824 |
| telegraphic_request | 0.868 | 0.856 | 0.012 | 9 | 6 | 0.607 |

This cap-corrected GLM probe does not show a statistically meaningful degradation on any single dimension. The result should be reported as a robustness/control finding, not as evidence that all capable models fail under these perturbations.

## Full-Pool Cap-1024 Paired Result

The same cap-corrected GLM setup was then run on the full 2,351-example frozen pool. The run used OpenRouter pinned to DeepInfra, temperature 0, and `REALISTIC_BFCL_ROUTER_MAX_OUTPUT_TOKENS=1024`.

| dimension | clean acc | noisy acc | drop | clean-to-noisy failures | noisy-to-clean fixes | net regressions | McNemar exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| typos | 0.845 | 0.840 | 0.006 | 49 | 36 | 13 | 0.193 |
| cursing | 0.845 | 0.831 | 0.014 | 76 | 43 | 33 | 0.003 |
| irrelevant_context | 0.845 | 0.838 | 0.007 | 52 | 36 | 16 | 0.109 |
| removed_spaces | 0.845 | 0.839 | 0.006 | 48 | 33 | 15 | 0.119 |
| argumentative_challenge | 0.845 | 0.837 | 0.009 | 60 | 40 | 20 | 0.057 |
| pasted_context_block | 0.845 | 0.843 | 0.002 | 46 | 42 | 4 | 0.749 |
| telegraphic_request | 0.845 | 0.825 | 0.020 | 86 | 38 | 48 | 0.000019 |

On the full pool, GLM-4.6 still does not show broad fragility across all seven dimensions. The clearest signal is `telegraphic_request`, and `cursing` is also statistically directional. `pasted_context_block` remains essentially null. This supports a narrower interpretation: stronger tool-calling models can absorb much of this deterministic messiness, but terse user shorthand still exposes a paired clean-success/noisy-failure surface.
