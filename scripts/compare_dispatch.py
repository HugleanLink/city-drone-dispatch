"""Reproduce the default dashboard comparison with a local source CSV.

Pass a clean checkout of the commit being measured. Write results outside the
repository when working with a private source file. The output omits coordinates.
"""
import argparse
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[201, 202, 203])
    parser.add_argument("--stress", action="store_true", help="302 orders with the 201-order staffing and fleet")
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    from src import batch, metrics
    from src.calibration import fit
    from src.data import day_groups, generate, prepare, profile_for_groups
    from src.efficiency import fleet_estimate, operating_settings
    from src.model import FlightRules
    from src.workspace import construct

    # Instrument old checkouts without changing their engine or its policy.
    if not hasattr(batch, "diagnostics"):
        current_src = Path(__file__).resolve().parents[1] / "src"
        metrics.resource_usage = load_module("src.research_metrics", current_src / "metrics.py").resource_usage
        probe = load_module("src.research_diagnostics", current_src / "diagnostics.py")
        old_metrics = batch.metrics

        def instrument(run):
            summary, _ = probe.diagnostics(run)
            stages = old_metrics(run)
            delivered = [r["actual"] - r["order"].arrival for r in run.records.values() if r["actual"] is not None]
            stages["p50_minutes"] = median(delivered) if delivered else None
            return {**stages, **summary}

        batch.metrics = instrument

    measured_metrics = batch.metrics

    def audit(run):
        for resource, slots in run.cal.slots.items():
            if any(a.end > b.start for a, b in zip(slots, slots[1:])):
                raise AssertionError(f"overlapping reservations for {resource}")
        dispatches = [item["id"] for item in run.log if item["type"] == "dispatch"]
        if len(dispatches) != len(set(dispatches)):
            raise AssertionError("an order was dispatched more than once")
        return measured_metrics(run)

    batch.metrics = audit
    profile = prepare(args.input)
    groups = day_groups(profile)
    cut = max(1, min(4, len(groups) - 1))
    planning = profile_for_groups(profile, groups[:cut])
    operation = profile_for_groups(profile, groups[cut:] or groups)
    ground = fit(planning)
    network = construct(planning, [], sha256(args.input.read_bytes()).hexdigest(), 4000, False,
                        no_replacement=True, rules=FlightRules(weight=2, order_min=1, order_max=4.04,
                        access_km=3, leg_min=.5, leg_max=6.06, report=True, exclude_group_meal=True),
                        max_hubs=15, max_spokes=10, max_routes=None)
    settings = operating_settings(operation, network, 201)
    settings["aircraft_by_hub"] = fleet_estimate(network, generate(operation, 201, 201))
    rows = []
    count = 302 if args.stress else 201
    print(f"{count} orders, {len(network.routes)} flight legs, {sum(settings['rider_hours'])} scheduled rider-hours",
          flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    def report(row):
        nonlocal started
        now = perf_counter()
        row["seconds"] = round(now - started, 3)
        started = now
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(row["seed"], row["strategy"], row["mean_minutes"], row["saved_minutes"], row["air"],
              row["saved_rider_minutes"], row["seconds"], flush=True)

    batch.run_batch(operation, network, args.seeds, ("normal",), settings=settings, orders_per_day=count,
                    ground_model=ground, include_orders=True, progress=report)
    print("Checked all reservation calendars and dispatch IDs", flush=True)


if __name__ == "__main__":
    main()
