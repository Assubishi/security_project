from pathlib import Path
import json
from statistics import mean

MODES = ["none", "ragdefender", "robustrag", "stacked_gated"]
SEEDS = ["seed7", "seed13"]
METRICS = [
    "attack_success_rate",
    "clean_accuracy",
    "poison_in_context_rate",
    "avg_latency_sec",
    "avg_llm_calls",
]

base = Path("results")
summary = {}

for mode in MODES:
    vals = {m: [] for m in METRICS}
    for seed in SEEDS:
        path = base / seed / mode / "metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for m in METRICS:
            vals[m].append(data[m])

    summary[mode] = {m: mean(v) for m, v in vals.items() if v}

print(f"{'mode':16} {'ASR':>8} {'clean_acc':>10} {'poison_ctx':>11} {'latency':>10} {'llm_calls':>10}")
for mode, row in summary.items():
    print(
        f"{mode:16} "
        f"{row['attack_success_rate']:8.4f} "
        f"{row['clean_accuracy']:10.4f} "
        f"{row['poison_in_context_rate']:11.4f} "
        f"{row['avg_latency_sec']:10.4f} "
        f"{row['avg_llm_calls']:10.4f}"
    )
