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
