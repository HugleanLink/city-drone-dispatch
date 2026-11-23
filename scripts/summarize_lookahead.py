"""Check experiment pairing and summarize the paired bootstrap by random seed."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from statistics import mean, median, stdev

POLICIES = ("ground", "fixed", "dynamic", "legacy", "final")
FIELDS = ("mean_minutes", "ground_mean_minutes", "ground_p90_minutes", "ground_mean_wait",
          "rider_minutes", "delivered_air_orders", "reused_ground_order_count", "seconds")
PAIR_INPUT = ("input_sha256", "network_sha256", "calibration", "arrivals", "rider_hours", "routes")


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def interval(values):
    rng = Random(90966)
    if len(values) < 2:
        return None
    distribution = sorted(mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [distribution[249], distribution[9749]]


def summarize(rows, variants):
    per_scenario = defaultdict(dict)
    for key, row in rows.items():
        scenario, seed, policy = key
        per_scenario[scenario].setdefault(seed, {})[policy] = row
    analysis = {}
    for scenario, seed_rows in sorted(per_scenario.items()):
        seeds = sorted(seed_rows)
        for policies in seed_rows.values():
            assert set(policies) == set(POLICIES)
            assert all(policies["ground"][key] == row[key] for row in policies.values() for key in PAIR_INPUT)
            assert all(row["completed"] == row["arrivals"] for row in policies.values())
        aggregate = {}
        for name in POLICIES:
            vals = [seed_rows[seed][name] for seed in seeds]
            aggregate[name] = {field: mean(row[field] for row in vals) for field in FIELDS}
            aggregate[name]["time_sd"] = stdev(row["mean_minutes"] for row in vals) if len(vals)>1 else None
            aggregate[name]["seconds_median"] = median(row["seconds"] for row in vals)
        deltas = {}
        for name in ("dynamic", "legacy"):
            d = {}
            for field in FIELDS[:5]:
                values = [seed_rows[seed][name][field]-seed_rows[seed]["final"][field] for seed in seeds]
                d[field] = dict(mean=mean(values), sd=stdev(values) if len(values)>1 else None,
                                interval_95=interval(values), positive=sum(v>0 for v in values),
                                negative=sum(v<0 for v in values))
            deltas[name] = d
        search = Counter()
        for seed in seeds:
            search.update(seed_rows[seed]["final"]["search"])
        analysis[scenario] = dict(seeds=seeds, averages=aggregate, final_savings=deltas, search=dict(search),
                                  released=sum(seed_rows[s]["final"]["released_air_order_count"] for s in seeds),
                                  reused=sum(seed_rows[s]["final"]["reused_ground_order_count"] for s in seeds),
                                  reused_vs_dynamic=sum(seed_rows[s]["final"]["reused_vs_dynamic_count"] for s in seeds))
    ablation = defaultdict(dict)
    for row in variants:
        scenario, seed, name = (row[k] for k in ("scenario","seed","policy"))
        ref = rows[scenario,seed,"dynamic"]
        assert all(ref[k] == row[k] for k in PAIR_INPUT) and row["completed"] == ref["completed"]
        ablation[scenario].setdefault(name, []).append(row)
    for scenario in ablation:
        entries = ablation[scenario]
        seeds = sorted({r["seed"] for group in entries.values() for r in group})
        entries["final8"] = [rows[scenario,seed,"final"] for seed in seeds]
        for key, group in list(entries.items()):
            group.sort(key=lambda r:r["seed"])
            assert [row["seed"] for row in group] == seeds
            entries[key] = {"seeds": seeds,
                            **{f"{field}_mean": mean(r[field] for r in group) for field in
                               ("mean_minutes", "ground_mean_wait", "ground_p90_minutes", "rider_minutes", "seconds")},
                            "active_wait_minutes": sum(r["search"].get("active_wait_minutes",0) for r in group)}
    return {"scenarios": analysis, "ablations": dict(ablation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--variants", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, paired = load(args.baseline), load(args.paired)
    rows = {(r["scenario"], r["seed"], r["policy"]): r for r in baseline}
    for row in paired:
        key = tuple(row[label] for label in ("scenario","seed","policy"))
        if key in rows:
            old = rows[key]
            for field in (*PAIR_INPUT, "completed", "mean_minutes", "ground_mean_wait", "ground_p90_minutes",
                          "ground_mean_minutes", "rider_minutes", "delivered_air_orders", "search"):
                assert old[field] == row[field], (key, field)
        else:
            rows[key] = row
    variants = [r for path in args.variants for r in load(path)]
    report = summarize(rows, variants)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = []
    for _, row in sorted(rows.items()):
        flat.append({**{k: v for k, v in row.items() if not isinstance(v, (list, dict))},
                     **{f"search_{k}": v for k, v in row["search"].items()}})
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with (args.output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
