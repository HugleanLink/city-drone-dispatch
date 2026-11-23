"""Run frozen dispatch inputs against a specified checkout and policy variant."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")


def initialize(checkout, context):
    global FROZEN
    sys.path.insert(0, str(checkout))
    FROZEN = context


def cases(base, base_slots, trial, trial_slots, records):
    """Count distinct verified beneficiary orders, without claiming causal isolation."""
    old_direct = defaultdict(list)
    occupations = defaultdict(list)
    for slot in trial_slots:
        occupations[slot.order, slot.resource].append(slot)
    for slot in base_slots:
        own = trial[slot.order]
        if slot.activity == "地面直送" and own["mode"] == "air" and own["actual"] is not None:
            old_direct[slot.resource].append(slot)
    reused = {}
    released = set()
    for rider, slots in old_direct.items():
        for slot in slots:
            remaining = slot.end - slot.start - sum(
                max(0, min(slot.end, other.end) - max(slot.start, other.start))
                for other in occupations[slot.order, rider])
            if remaining:
                released.add(slot.order)
    for slot in trial_slots:
        key = slot.order
        current, old = trial[key], base[key]
        if (key in reused or slot.activity != "地面直送" or not current["ground_only"]
                or old["actual"] is None or current["actual"] is None or old["actual"] <= current["actual"]
                or old["first"] <= current["first"]):
            continue
        for free in old_direct[slot.resource]:
            start, end = max(free.start, slot.start), min(free.end, slot.end)
            if start >= end or any(q.start < end and start < q.end for q in occupations[free.order, slot.resource]):
                continue
            reused[key] = dict(air_order=free.order, ground_order=key, rider=slot.resource, start=start, end=end,
                               ground_start_before=old["first"], ground_start_after=current["first"],
                               ground_delivery_before=old["actual"], ground_delivery_after=current["actual"],
                               ground_pickup_after=records[key].get("fetch"))
            break
    return len(released), list(reused.values())


def plan_summary(plan):
    if plan is None:
        return None
    return {"mode": plan.mode, "route": plan.route, "first": plan.first, "eta": plan.eta,
            "slots": [asdict(slot) for slot in plan.slots]}


def trace_class():
    from src.sim import Engine

    class Traced(Engine):
        def __init__(self, *args, trace=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit_trace = trace
            self.trace_rows = []
            self.pending_trace = None
            self.diagnostic_counts = Counter()
            self.histogram = Counter()
            self.explicit = defaultdict(set)

        def evaluate(self, order, others=(), now=None, *, cancelled=lambda: False):
            guide, branch = super().evaluate(order, others, now, cancelled=cancelled)
            if guide is None:
                return guide, branch
            self.diagnostic_counts["legacy_rollouts"] += 1
            self.histogram[min(len(others), 12)] += 1
            if branch["mode"] == "wait":
                self.diagnostic_counts["legacy_wait_votes"] += 1
                self.diagnostic_counts["legacy_wait_with_alternate"] += branch["alternative"] is not None
                if branch["alternative"] is None:
                    self.explicit[order.id].add(self.now)
            if self.audit_trace:
                immediate = self.opts(order, now=max(self.now, order.ready), force=True)
                alternatives = branch.get("alternative")
                self.pending_trace = row = {"time": self.now, "pivot": order.id,
                    "candidate_orders": [o.id for o in sorted(others, key=lambda o: (o.ready, o.id))[:12]],
                    "branches": guide["branches"], "locally_fastest": plan_summary(self._best(immediate)),
                    "choice": branch["mode"], "virtual_first": plan_summary(branch["plan"] or
                        (alternatives[1] if alternatives else None))}
                self.trace_rows.append(row)
                if self.diagnostic_counts["full_route_trace"] < 8 and (
                        branch["mode"] == "wait" or (branch["plan"] and self._best(immediate) != branch["plan"])):
                    self.diagnostic_counts["full_route_trace"] += 1
                    row["focal_ground"] = plan_summary(immediate[0])
                    row["focal_routes"] = {rid: plan_summary(self._air_route(order, self.routes[rid], self.cal,
                        self.now, True, None, cancelled)) for rid in order.candidate_route_ids}
            return guide, branch

        def _commit(self, order, plan, cancelled=lambda: False):
            completed = super()._commit(order, plan, cancelled)
            if completed and self.audit_trace:
                row = self.pending_trace or {"time": self.now}
                row.update(committed_order=order.id, committed=plan_summary(plan))
                if self.pending_trace is None:
                    self.trace_rows.append(row)
                self.pending_trace = None
            return completed

        def dispatch(self, cancelled=lambda: False):
            self.pending_trace = None
            return super().dispatch(cancelled)

    return Traced


def run_group(job):
    label, seed, names, trace = job
    import numpy as np
    from src.batch import digest
    from src.calibration import GroundTimes
    from src.diagnostics import diagnostics
    from src.efficiency import changes, details
    from src.metrics import metrics

    network, orders, settings, event, calibration = FROZEN[label, seed]
    ground_model = GroundTimes.from_dict(calibration)
    net_hash = digest(asdict(network))
    fingerprint = digest({"orders": [o.input_dict() for o in orders], "event": event,
                          "disturbance": None, "settings": settings, "network": asdict(network),
                          "calibration": ground_model.version})
    output, base, dynamic = [], None, None
    engine = trace_class()
    for name in names:
        strategy = name if name in ("ground", "fixed", "dynamic") else "lookahead"
        started = perf_counter()
        run = engine(network, orders, strategy, settings, ground_model=ground_model, trace=trace)
        if name == "legacy" and hasattr(run, "lookahead_policy"):
            run.lookahead_policy = None
        elif name not in ("ground", "fixed", "dynamic", "legacy"):
            from src.lookahead import research_policy
            run.lookahead_policy = research_policy(name)
            if trace:
                run.lookahead_trace = []
        if event:
            run.limit(*event)
        run.finish()
        elapsed = perf_counter() - started
        for resource, slots in run.cal.slots.items():
            assert all(a.end <= b.start for a, b in zip(slots, slots[1:])), resource
        assigned = [item["id"] for item in run.log if item["type"] == "dispatch"]
        assert len(assigned) == len(set(assigned))
        diag, per_order = diagnostics(run)
        trial = details(run)
        if name == "ground":
            base = trial
        if name == "dynamic":
            dynamic = trial
        ground_times = [r["actual"]-r["arrival"] for r in trial[0].values()
                        if r["ground_only"] and r["actual"] is not None]
        ground_wait = [r["wait"] for r in trial[0].values() if r["ground_only"] and r["actual"] is not None]
        result = {"scenario": label, "seed": seed, "policy": name, "seconds": elapsed, "input_sha256": fingerprint,
                  "network_sha256": net_hash, "calibration": ground_model.version,
                  "rider_hours": sum(settings["rider_hours"]), "routes": len(network.routes),
                  **metrics(run), **{k:v for k,v in diag.items() if not isinstance(v, list)},
                  "ground_mean_minutes": mean(ground_times) if ground_times else None,
                  "ground_p90_minutes": float(np.percentile(ground_times, 90)) if ground_times else None,
                  "ground_mean_wait": mean(ground_wait) if ground_wait else None,
                  "policy_diagnostics": dict(run.diagnostic_counts), "candidate_histogram": dict(run.histogram),
                  "legacy_explicit_wait_pivot_minutes": sum(len(x) for x in run.explicit.values()),
                  "legacy_explicit_wait_pivots": len(run.explicit),
                  "search": getattr(run, "lookahead_stats", {})}
        if base is not None:
            result.update(changes(base, trial, include_case=False))
            released, reused = cases(base[0], base[1], trial[0], trial[1], run.records)
            result.update(released_air_order_count=released, reused_ground_order_count=len(reused),
                          reuse_examples=reused[:5])
        if dynamic is not None:
            result["vs_dynamic"] = changes(dynamic, trial, include_case=False)
            released, reused = cases(dynamic[0], dynamic[1], trial[0], trial[1], run.records)
            result.update(released_vs_dynamic_count=released, reused_vs_dynamic_count=len(reused),
                          reuse_vs_dynamic_examples=reused[:5])
        if trace:
            result["trace"] = run.trace_rows
            result["order_times"] = {key:{**row, **per_order[key], "fetch": run.records[key].get("fetch")}
                                     for key,row in trial[0].items()}
            result["slots"] = [asdict(slot) for group in run.cal.slots.values() for slot in group]
            result["explicit_wait_minutes"] = {key: sorted(val) for key, val in run.explicit.items()}
            result["lookahead_trace"] = getattr(run, "lookahead_trace", [])
        output.append(result)
    return output


def make_inputs(input_path, group_specs):
    from src.calibration import fit
    from src.data import day_groups, generate, prepare, profile_for_groups
    from src.efficiency import fleet_estimate, operating_settings
    from src.model import FlightRules
    from src.workspace import construct
    profile = prepare(input_path)
    groups = day_groups(profile)
    planning = profile_for_groups(profile, groups[:4])
    operation = profile_for_groups(profile, groups[4:])
    calibration = fit(planning).to_dict()
    network = construct(planning, [], sha256(input_path.read_bytes()).hexdigest(), 4000, False,
                        no_replacement=True, rules=FlightRules(weight=2, order_min=1, order_max=4.04,
                            access_km=3, leg_min=.5, leg_max=6.06, report=True, exclude_group_meal=True),
                        max_hubs=15, max_spokes=10, max_routes=None)
    staffing = operating_settings(operation, network, 201)
    staffing["aircraft_by_hub"] = fleet_estimate(network, generate(operation, 201, 201))
    busiest = max(staffing["aircraft_by_hub"], key=lambda h: (sum(r.coverage for r in network.routes if r.hub == h), h))
    # The one-route network preserves the geographical extent and rider start locations.
    single = replace(network, routes=(max(network.routes, key=lambda r: (r.coverage, r.id)),))
    data = {}
    for label, seed in group_specs:
        count = 302 if label in ("stress", "stress_relief") else 201
        orders = generate(operation, count, seed)
        if count > 201:
            assert orders[:201] == generate(operation, 201, seed)
        setting = dict(staffing)
        if label in ("relief", "stress_relief"):
            setting["rider_hours"] = [x + (x > 0) for x in staffing["rider_hours"]]
        if label == "tight":
            setting["rider_hours"] = [max(1, x-1) if x else 0 for x in staffing["rider_hours"]]
        event = ("aircraft", busiest, staffing["aircraft_by_hub"][busiest], 660, 780) if label == "hub" else None
        event = (*event, None) if event else None
        data[label, seed] = (single if label == "single" else network, orders, setting, event, calibration)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", nargs="+", default=["normal:201"], help="scenario:start[:end], inclusive")
    parser.add_argument("--policy", nargs="+", default=["ground", "fixed", "dynamic", "legacy"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    groups = []
    for group in args.group:
        name, first, *end = group.split(":")
        groups.extend((name, seed) for seed in range(int(first), int(end[0] if end else first)+1))
    frozen = make_inputs(args.input, groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(label, seed, args.policy, args.trace) for label, seed in groups]
    with args.output.open("w", encoding="utf-8") as stream:
        if args.workers == 1:
            initialize(checkout, frozen)
            for job in jobs:
                write_rows(stream, run_group(job))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(checkout, frozen)) as pool:
                for future in as_completed([pool.submit(run_group, job) for job in jobs]):
                    write_rows(stream, future.result())


def write_rows(stream, rows):
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(row["scenario"], row["seed"], row["policy"], round(row["mean_minutes"], 2),
              round(row["seconds"], 2), row["completed"], flush=True)
    stream.flush()


if __name__ == "__main__":
    main()
