import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import ceil

from .data import clock_window, generate, profile_for_groups, sources
from .metrics import metrics
from .efficiency import changes, details, fleet_estimate, operating_settings
from .model import LAST_ARRIVAL, START, STRATEGIES, match_order
from .sim import Engine
from .storage import encode


SCENARIOS = {
    "normal": "常态", "heavy": "高需求", "battery": "备用电池受限",
    "aircraft": "机队受限", "pad": "起降位受限", "demand": "局部订单激增",
    "pads2": "双起降位", "riders3": "减少骑手",
}
DISTURBANCES = {"battery": "备用电池", "aircraft": "无人机", "pad": "起降位", "demand": "运营订单量"}


@dataclass(frozen=True)
class Disturbance:
    kind: str
    site: str
    percent: int
    start: int
    end: int

    def __post_init__(self):
        if self.kind not in DISTURBANCES:
            raise ValueError("未知扰动类型")
        latest = self.start + LAST_ARRIVAL
        if not START <= self.start < LAST_ARRIVAL or not self.start < self.end <= latest:
            raise ValueError("请检查扰动起止时间")
        if self.kind == "demand" and (not -100 <= self.percent <= 200 or not self.percent):
            raise ValueError("运营订单量变化须为 -100% 至 200% 的非零值")
        if self.kind != "demand" and not 1 <= self.percent <= 100:
            raise ValueError("资源停用比例须为 1% 至 100%")


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def target_hub(network, planning):
    links = {r.id: r.hub for r in network.routes}
    counts = {r.hub: 0 for r in network.routes}
    for order in planning:
        route = match_order(network, order).route_id
        if route:
            counts[links[route]] += 1
    return min(counts, key=lambda hub: (-counts[hub], hub), default=None)


def resolve_restriction(run, disturbance):
    ids = run.resources(disturbance.kind, disturbance.site)
    if not ids:
        raise ValueError("航网中没有所选站点或资源")
    count = ceil(len(ids) * disturbance.percent / 100)
    return disturbance.kind, disturbance.site, count, disturbance.start, disturbance.end, ids[:count]


def adjust_demand(orders, profile, seed, disturbance):
    eligible = [o for o in orders if clock_window(o.arrival, disturbance.start, disturbance.end)]
    goal = ceil(len(eligible) * (100 + disturbance.percent) / 100)
    if goal < len(eligible):
        removed = {o.id for o in sorted(eligible, key=lambda o: o.id, reverse=True)[:len(eligible) - goal]}
        return [o for o in orders if o.id not in removed]
    return orders + generate(profile, goal - len(eligible), seed + 100_000,
                             window=(disturbance.start, disturbance.end), prefix="D")


def run_batch(profile, network, seeds, scenarios, *, hub=None, settings=None, orders_per_day=None,
              disturbance=None, cancelled=lambda: False, progress=lambda row: None, ground_model=None, include_orders=False):
    if orders_per_day is not None and not 1 <= orders_per_day <= 100_000:
        raise ValueError("每轮仿真订单数须为 1 至 100,000")
    output = []
    network_data = asdict(network)
    network_hash = digest(network_data)
    for seed in seeds:
        for scenario in scenarios:
            if cancelled():
                return output
            heavy = scenario != "normal"
            count = orders_per_day if orders_per_day is not None else (302 if heavy else 201)
            orders = generate(profile, count, seed, "heavy" if heavy else "operation")
            run_settings = {**(settings or {})}
            if scenario == "pads2":
                run_settings["pads"] = 2
            if scenario == "riders3":
                run_settings["riders"] = 3

            if ground_model and settings is None:
                operation = profile_for_groups(profile, sources(profile)["operation"])
                defaults = operating_settings(operation, network, count)
                defaults["aircraft_by_hub"] = fleet_estimate(network, orders)
                if scenario == "riders3":
                    defaults.pop("rider_hours")
                run_settings = {**defaults, **run_settings}
            prototype = Engine(network, settings=run_settings, ground_model=ground_model)
            selected_hub = hub or min(prototype.hubs, default=None)
            event = None
            applied = disturbance
            if applied is None and scenario == "demand":
                applied = Disturbance("demand", "", 50, 720, 750)
            if applied and applied.kind == "demand":
                orders = adjust_demand(orders, profile, seed, applied)
            elif applied:
                event = resolve_restriction(prototype, applied)
            elif scenario in ("battery", "aircraft", "pad"):
                if not selected_hub:
                    raise ValueError("资源受限场景需要至少一个启用 Hub")
                route = next(r for r in network.routes if r.hub == selected_hub)
                site = (route.hub_site or route.hub) if scenario == "pad" else selected_hub
                capacity, start, end = {"battery": (3, 720, 740), "aircraft": (2, 720, 740),
                                        "pad": (1, 730, 745)}[scenario]
                ids = prototype.resources(scenario, site)[:capacity]
                if len(ids) != capacity:
                    raise ValueError(f"场景 {scenario} 的初始资源不足")
                event = scenario, site, capacity, start, end, ids

            applied_data = asdict(applied) if applied else None
            input_hash = digest({"orders": [asdict(o) for o in orders], "event": event,
                                 "disturbance": applied_data, "settings": run_settings, "network": network_data,
                                 "calibration": ground_model.version if ground_model else None})
            baseline = None
            for strategy in STRATEGIES:
                if cancelled():
                    return output
                run = Engine(network, orders, strategy, run_settings, ground_model=ground_model)
                if event:
                    run.limit(*event)
                run.finish(cancelled)
                if cancelled():
                    return output
                row = {"seed": seed, "scenario": scenario, "strategy": strategy, "input_sha256": input_hash,
                       "network_sha256": network_hash, "base_orders": count, "disturbance": applied_data,
                       **metrics(run)}
                trial = details(run)
                if baseline is None:
                    baseline = trial
                row.update(changes(baseline, trial, include_case=strategy != "ground"))
                row["undelivered"] = row["arrivals"] - row["completed"]
                row["rider_breakdown"] = trial[2]
                if include_orders:
                    row["orders"] = trial[0]
                row["calibration"] = ground_model.version if ground_model else None
                row["settings"] = run_settings
                output.append(row)
                progress(row)
    return output


def load_results(payload, csv_file=False):
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError("对照结果超过 20 MB")
    rows = (list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))) if csv_file
            else json.loads(payload))
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10_000:
        raise ValueError("对照结果需要 1 至 10,000 条策略记录")
    validated, seen = [], set()
    group_hashes = {}
    for row in rows:
        if row["scenario"] not in SCENARIOS or row["strategy"] not in STRATEGIES:
            raise ValueError("对照结果包含未知场景或策略")
        if not re.fullmatch(r"[a-f0-9]{64}", row["input_sha256"]):
            raise ValueError("对照结果缺少有效的共同输入哈希")
        network = row.get("network_sha256") or ""
        if network and not re.fullmatch(r"[a-f0-9]{64}", network):
            raise ValueError("对照结果的航网版本无效")
        numbers = {key: int(row[key]) for key in ("seed", "arrivals", "on_time", "ground_only", "ground_on_time")}
        total, timely, ground, ground_timely = (numbers[key] for key in
                                              ("arrivals", "on_time", "ground_only", "ground_on_time"))
        if not (0 <= ground_timely <= ground <= total and ground_timely <= timely <= total
                and timely - ground_timely <= total - ground):
            raise ValueError("准时订单数与分母不一致")
        group = network, row["scenario"], numbers["seed"]
        if group_hashes.setdefault(group, row["input_sha256"]) != row["input_sha256"]:
            raise ValueError("同一场景和种子的策略输入哈希不一致")
        item_id = (*group, row["strategy"])
        if item_id in seen:
            raise ValueError("对照结果有重复的策略记录")
        seen.add(item_id)
        validated.append({**row, **numbers, "network_sha256": network,
                          "on_time_rate": timely / total if total else None,
                          "ground_on_time_rate": ground_timely / ground if ground else None})
    return validated


def summarize_results(rows):
    pairs = {}
    buckets = defaultdict(list)
    for row in rows:
        group = row.get("network_sha256", ""), row["scenario"]
        buckets[(*group, row["strategy"])].append(row)
        if row["strategy"] == "ground":
            pairs[(*group, row["seed"], row["input_sha256"])] = row
    summary = []
    for (network, scenario, strategy), items in sorted(
            buckets.items(), key=lambda entry: (*entry[0][:2], list(STRATEGIES).index(entry[0][2]))):
        result = {"network_sha256": network, "scenario": scenario, "strategy": strategy, "days": len(items)}
        for numerator, denominator, metric in (("on_time", "arrivals", "on_time_rate"),
                                                ("ground_on_time", "ground_only", "ground_on_time_rate")):
            total = sum(row[denominator] for row in items)
            timely = sum(row[numerator] for row in items)
            differences = []
            for row in items:
                other = pairs.get((network, scenario, row["seed"], row["input_sha256"]))
                if other and row[metric] is not None and other[metric] is not None:
                    differences.append(100 * (row[metric] - other[metric]))
            result.update({numerator: timely, denominator: total, metric: timely / total if total else None,
                           f"{metric}_delta": sum(differences) / len(differences) if differences else None,
                           f"{metric}_paired_days": len(differences)})
        summary.append(result)
    return summary


def write_results(rows, path):
    if not rows:
        return
    fields = [k for k, v in rows[0].items() if not isinstance(v, (dict, list)) and k != "disturbance"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
