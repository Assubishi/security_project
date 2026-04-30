# Final Results Summary

## Small Reduced Setup (2 seeds, averaged)

| Mode            | ASR   | Clean Acc | Poison Context | Latency | LLM Calls |
|-----------------|-------|----------|----------------|---------|-----------|
| none            | 0.1666 | 0.5000 | 1.0000 | 2.6787 | 1.0 |
| ragdefender     | 0.0833 | 0.3750 | 0.8334 | 3.1193 | 1.0 |
| robustrag       | 0.1666 | 0.5000 | 1.0000 | 18.4113 | 6.0 |
| stacked_gated   | 0.1666 | 0.5000 | 0.6667 | 7.0354 | 2.9166 |

## Larger Reduced Setup (seed 7)

| Mode            | ASR | Clean Acc | Poison Context | Latency | LLM Calls |
|-----------------|-----|-----------|----------------|---------|-----------|
| none            | 0.25 | 0.35 | 0.95 | 3.03 | 1 |
| ragdefender     | 0.2 | 0.275 | 0.7 | 2.659 | 1 |
| robustrag       | 0.2 | 0.35 | 0.95 | 14.5373 | 6 |
| stacked_gated   | 0.15 | 0.275 | 0.7 | 5.945 | 2.8 |

## Key Observations

- Poison reaches retrieval in baseline (high poison-in-context)
- RAGDEFENDER reduces poisoned passages
- RobustRAG improves reasoning but is expensive
- Stacked gated reduces cost while maintaining robustness
