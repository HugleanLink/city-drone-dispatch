"""Capacity estimates and paired efficiency measures."""
from collections import Counter, defaultdict
from math import ceil

import numpy as np

from .model import match_order, minutes


def operating_settings(profile, network, count):
    rows = profile.rows
    weights = rows["count"].to_numpy(dtype=float)
    per_hour = np.bincount(rows.minute.to_numpy(dtype=int) // 60, weights=weights, minlength=24)
    forecast = per_hour * (count / weights.sum())
    riders = [ceil(max(0, value) / 3) for value in forecast]
    routes = {route.id: route for route in network.routes}
    return {"rider_hours": riders, "aircraft_by_hub": {route.hub: 1 for route in routes.values()}}


def fleet_estimate(network, orders):
    routes = {route.id: route for route in network.routes}
    buckets = Counter()
    durations = defaultdict(list)
    for order in orders:
        route = routes.get(match_order(network, order).route_id)
        if route:
            buckets[route.hub, order.arrival // 60] += 1
            durations[route.hub].append(2 * minutes(route.km, 40) + 8)
    return {hub: max(1, ceil(max((buckets[hub, hour] for hour in range(24)), default=0)
                             * np.median(durations.get(hub, [0])) / (60 * .8)))
            for hub in sorted({route.hub for route in routes.values()})}


def details(run):
    orders = {}
    for key, record in run.records.items():
        if record["status"] in ("removed", "scheduled"):
            continue
        order = record["order"]
        orders[key] = {"arrival": order.arrival, "actual": record["actual"], "first": record.get("first"),
                       "wait": record["first"] - order.ready if record.get("first") is not None else None,
                       "ground_only": order.ground_only, "mode": record["mode"]}
    slots = [slot for resource, slots in run.cal.slots.items() if resource.startswith("rider:") for slot in slots]
    breakdown = Counter()
    for slot in slots:
        zone = slot.resource.split(":")[1]
        for hour in range(slot.start // 60, (slot.end-1) // 60 + 1):
            overlap = min(slot.end, (hour+1)*60) - max(slot.start, hour*60)
            breakdown[hour, zone, slot.activity] += overlap
    return orders, slots, [dict(hour=h, zone=z, activity=a, minutes=n) for (h,z,a),n in sorted(breakdown.items())]


def changes(baseline, trial, *, include_case=True):
    base_orders, base_slots, _ = baseline
    orders, slots, _ = trial
    paired = [key for key in base_orders.keys() & orders.keys()
              if base_orders[key]["actual"] is not None and orders[key]["actual"] is not None]
    base = [base_orders[key]["actual"] - base_orders[key]["arrival"] for key in paired]
    current = [orders[key]["actual"] - orders[key]["arrival"] for key in paired]
    ground = [key for key in paired if orders[key]["ground_only"]]
    waits = [base_orders[key]["wait"] - orders[key]["wait"] for key in ground]
    old_tail = [base_orders[key]["actual"]-base_orders[key]["arrival"] for key in ground]
    new_tail = [orders[key]["actual"]-orders[key]["arrival"] for key in ground]
    saved = sum(base) - sum(current)
    return dict(paired_orders=len(paired), paired_baseline_minutes=sum(base), paired_strategy_minutes=sum(current),
                saved_minutes=saved / len(base) if base else None,
                time_reduction=saved / sum(base) if sum(base) else None,
                paired_ground=len(ground), saved_ground_wait=sum(waits) / len(waits) if waits else None,
                saved_ground_p90=float(np.percentile(old_tail, 90) - np.percentile(new_tail, 90)) if ground else None,
                saved_rider_minutes=sum(s.end-s.start for s in base_slots) - sum(s.end-s.start for s in slots),
                example=shared_rider_case(base_orders, orders, base_slots, slots) if include_case else None)


def shared_rider_case(base, trial, base_slots, slots):
    released = defaultdict(list)
    for slot in base_slots:
        if (slot.activity == "地面直送" and trial[slot.order]["mode"] == "air"
                and trial[slot.order]["actual"] is not None):
            released[slot.resource].append(slot)
    kept = defaultdict(list)
    for slot in slots:
        if slot.order in trial:
            kept[slot.order, slot.resource].append(slot)
    for slot in slots:
        key = slot.order
        if (slot.activity != "地面直送" or not trial[key]["ground_only"]
                or base[key]["actual"] is None or trial[key]["actual"] is None
                or base[key]["first"] <= trial[key]["first"] or base[key]["actual"] <= trial[key]["actual"]):
            continue
        for old in released[slot.resource]:
            start, end = max(old.start, slot.start), min(old.end, slot.end)
            if end <= start:
                continue
            if any(s.start < end and start < s.end for s in kept[old.order, slot.resource]):
                continue
            return dict(flyable=old.order, ground_only=key, rider=slot.resource, start=start, end=end,
                        earlier_start=base[key]["first"]-trial[key]["first"],
                        earlier_delivery=base[key]["actual"]-trial[key]["actual"])
    return None
