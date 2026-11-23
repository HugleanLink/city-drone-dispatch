"""Stage, rider and hub diagnostics for a completed simulation."""
from collections import Counter, defaultdict
from statistics import mean

from .metrics import resource_usage

STAGES = {
    "ready_wait": "支付至备妥",
    "dispatch_wait": "备妥至开始派送",
    "first_empty": "首段骑手空驶",
    "first_pickup": "首次取货交接",
    "ground_loaded": "地面载货配送",
    "first_loaded": "首段载货接驳",
    "hub_queue": "到达 Hub 至起飞",
    "flight_to_spoke": "起飞至航货可取",
    "spoke_queue": "航货可取至末段交接",
    "last_loaded": "末段交接及载货接驳",
    "last_empty": "末段骑手空驶（可与前序阶段并行）",
    "aircraft_duty": "飞机完整占用（含返航）",
    "battery_cycle": "电池绑定及周转（有释放时间时）",
}
COMMON = ("ready_wait", "dispatch_wait", "first_empty", "first_pickup")
MODE_STAGES = {
    "ground": (*COMMON, "ground_loaded"),
    "air": (*COMMON, "first_loaded", "hub_queue", "flight_to_spoke", "spoke_queue", "last_loaded",
            "last_empty", "aircraft_duty", "battery_cycle"),
}
NON_ADDITIVE = {"last_empty", "aircraft_duty", "battery_cycle"}


def _stages(record, slots, horizon):
    order = record["order"]
    first_name = "首段接驳" if record["mode"] == "air" else "地面直送"
    first = slots[first_name]
    values = {
        "ready_wait": order.ready - order.arrival,
        "dispatch_wait": record["first"] - order.ready,
        "first_empty": first.start - record["first"],
        "first_pickup": record["fetch"] - first.start,
    }
    if record["mode"] == "ground":
        return {**values, "ground_loaded": record["actual"] - record["fetch"]}
    last = slots["末段接驳"]
    hub, launch, ready, transfer = record.get("air_times") or (
        first.end, slots["载货起飞"].start, slots["降落、卸货及返程起飞"].start + 2,
        max(last.start, slots["降落、卸货及返程起飞"].start + 2),
    )
    aircraft = slots["航空任务"]
    empty = slots.get(("空驶", last.resource, last.start))
    battery = slots.get("绑定与周转")
    return {**values, "first_loaded": hub - record["fetch"], "hub_queue": launch - hub,
            "flight_to_spoke": ready - launch, "spoke_queue": transfer - ready,
            "last_loaded": record["actual"] - transfer, "last_empty": empty.end - empty.start if empty else 0,
            "aircraft_duty": aircraft.end - aircraft.start,
            "battery_cycle": battery.end - battery.start if battery and battery.end <= horizon else None}


def _hourly_riders(run, dispatched_at):
    used, tasks, overtime, held = Counter(), Counter(), Counter(), Counter()
    for rider in sorted(run.cal.tails):
        slots = run.cal.slots.get(rider, ())
        previous_end = 0
        previous_slot = None
        for slot in slots:
            end = min(slot.end, run.now)
            if end <= slot.start:
                continue
            for hour in range(slot.start // 60, (end - 1) // 60 + 1):
                minutes = min(end, (hour + 1) * 60) - max(slot.start, hour * 60)
                used[hour] += minutes
                if run.rider_hours is not None and run.rider_index[rider] > run.rider_hours[hour % 24]:
                    overtime[hour] += minutes
            if slot.activity != "空驶":
                start = (previous_slot.start if previous_slot and previous_end == slot.start and previous_slot.order == slot.order
                         and previous_slot.activity == "空驶" else slot.start)
                tasks[start // 60] += 1
            if slot.activity == "末段接驳" or slot.activity == "空驶":
                # This idle interval is blocked by an already committed future
                # task in the append-only rider schedule.
                left = max(previous_end, dispatched_at.get(slot.order, slot.start))
                if slot.start > left:
                    for hour in range(left // 60, (slot.start - 1) // 60 + 1):
                        held[hour] += min(slot.start, (hour + 1) * 60) - max(left, hour * 60)
            previous_end = end
            previous_slot = slot
    result = []
    for hour in range((run.now + 59) // 60):
        riders = run.rider_hours[hour % 24] if run.rider_hours is not None else len(run.cal.tails)
        scheduled = riders * min(60, run.now - hour * 60)
        available = scheduled + overtime[hour]
        result.append(dict(hour=hour, scheduled_riders=riders, scheduled_minutes=scheduled,
                           worked_minutes=used[hour], tasks_started=tasks[hour],
                           carryover_minutes=overtime[hour], reserved_idle_minutes=held[hour],
                           utilization=used[hour] / available if available else None))
    return result


def _hubs(run):
    pads = defaultdict(set)
    for route in run.routes.values():
        pads[route.hub].update(run.pads[route.hub_site or route.hub])
    result = []
    for hub, resources in sorted(run.hubs.items()):
        groups = {key: resources[key] for key in ("aircraft", "swap", "charge")}
        groups["pad"] = sorted(pads[hub])
        groups["battery"] = [*resources["battery"], *resources["mounted"].values()]
        usage = {key: resource_usage(run, set(ids)) for key, ids in groups.items()}
        duties = [s for plane in resources["aircraft"] for s in run.cal.slots.get(plane, ())
                  if s.activity == "航空任务" and s.start < run.now]
        result.append(dict(hub=hub, tasks=len(duties), resources=usage))
    return result


def diagnostics(run):
    by_order = defaultdict(dict)
    routes = {}
    for slots in run.cal.slots.values():
        for slot in slots:
            if slot.activity == "空驶":
                by_order[slot.order][slot.activity, slot.resource, slot.end] = slot
            else:
                by_order[slot.order][slot.activity] = slot
    dispatched_at = {e["id"]: e["minute"] for e in run.log if e["type"] == "dispatch"}
    by_stage = defaultdict(list)
    examples = []
    eligible = ground = air = switched = 0
    for key, record in run.records.items():
        if record["status"] in ("removed", "scheduled"):
            continue
        order = record["order"]
        eligible += not order.ground_only
        mode = record["mode"]
        execution = record.get("executed_route_id", order.route_id if mode == "air" else None)
        stages = _stages(record, by_order[key], run.end) if record["actual"] is not None else {}
        routes[key] = {"preferred_route_id": order.route_id, "executed_route_id": execution,
                       "candidate_route_ids": list(getattr(order, "candidate_route_ids", ()) or
                                                   ((order.route_id,) if order.route_id else ())),
                       "route_comparison": record.get("route_comparison"), "stages": stages}
        if record["actual"] is None:
            continue
        ground += mode == "ground"
        air += mode == "air"
        switched += execution is not None and execution != order.route_id
        for name, value in stages.items():
            if value is not None:
                by_stage[mode, name].append(value)
        case = record.get("route_comparison")
        if case and case["minutes_saved"] is not None and case["minutes_saved"] > 0:
            examples.append({"order": key, **case})

    hourly = _hourly_riders(run, dispatched_at)
    aggregate = {}
    for mode, values in MODE_STAGES.items():
        for name in values:
            samples = by_stage[mode, name]
            aggregate[f"{mode}_{name}_mean_minutes"] = mean(samples) if samples else None
    return (dict(air_eligible_orders=eligible, delivered_air_orders=air, delivered_ground_orders=ground,
                 reselected_route_orders=switched, rider_reserved_idle_minutes=sum(h["reserved_idle_minutes"] for h in hourly),
                 rider_carryover_minutes=sum(h["carryover_minutes"] for h in hourly), **aggregate,
                 rider_hourly=hourly, hub_resources=_hubs(run),
                 route_examples=sorted(examples, key=lambda e: (-e["minutes_saved"], e["order"]))[:5]), routes)


def diagnostic_tables(rows):
    tables = {name: [] for name in ("rider_hourly", "hub_resources", "order_diagnostics")}
    for row in rows:
        keys = {name: row[name] for name in ("seed", "scenario", "strategy")}
        tables["rider_hourly"].extend({**keys, **item} for item in row["rider_hourly"])
        tables["hub_resources"].extend({**keys, "hub": item["hub"], "tasks": item["tasks"],
                                         "resource": resource, **usage}
                                        for item in row["hub_resources"]
                                        for resource, usage in item["resources"].items())
        for order, item in row.get("orders", {}).items():
            comparison = item.get("route_comparison") or {}
            detail = {k: v for k, v in item.items() if k not in ("stages", "candidate_route_ids", "route_comparison")}
            tables["order_diagnostics"].append({**keys, "order": order, **detail,
                "candidate_route_ids": ",".join(item.get("candidate_route_ids", ())),
                "preferred_eta_at_dispatch": comparison.get("preferred_eta"),
                "executed_eta_at_dispatch": comparison.get("executed_eta"),
                "route_eta_saving_minutes": comparison.get("minutes_saved"),
                **item.get("stages", {})})
    return tables
