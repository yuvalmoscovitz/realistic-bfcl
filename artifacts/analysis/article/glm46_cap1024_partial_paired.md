# GLM-4.6 1024-Cap Partial Paired Results

This file summarizes the interrupted GLM-4.6 paired run with `REALISTIC_BFCL_ROUTER_MAX_OUTPUT_TOKENS=1024`. Only `typos` and `cursing` completed. `irrelevant_context` is partial and should not be used as a final result.

| dimension | status | n | clean acc | noisy acc | drop | clean->noisy fail | noisy->clean fix | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| typos | complete | 250 | 0.868 | 0.864 | 0.004 | 7 | 6 | 1.000 |
| cursing | complete | 250 | 0.868 | 0.852 | 0.016 | 10 | 6 | 0.454 |
| irrelevant_context | partial_interrupted | 56 | 0.911 | 0.929 | -0.018 | 1 | 2 | 1.000 |

Interpretation: the completed dimensions show no statistically meaningful GLM degradation at the 1024-token cap. `cursing` has a small raw 1.6 point drop, but the paired flip count is 10 vs 6 and McNemar p=0.454.
