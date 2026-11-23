"""Rolling dispatch for the efficiency version of multi-order lookahead."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .model import Order

if TYPE_CHECKING:
    from .sim import Plan

REGRET_MINUTES = 12
WAIT_BUDGET = 4


@dataclass(frozen=True)
class Policy:
    window: int = 8
    protect_ground: bool = False
    event_wait: bool = False
    diversify: bool = False
    peer_first: bool = True


POLICY = Policy(protect_ground=True, event_wait=True, diversify=True)


def research_policy(name):
    variants = {
        "control": Policy(),
        "objective": Policy(protect_ground=True),
        "waiting": Policy(event_wait=True),
        "candidates": Policy(diversify=True),
        "final": POLICY,
        "noswap": replace(POLICY, peer_first=False),
        **{f"final{size}": replace(POLICY, window=size) for size in (4, 8, 12, 20)},
    }
    return variants[name]


@dataclass(frozen=True)
class Action:
    order: Order
    plan: Plan | None
    kind: str
    until: int | None = None


@dataclass(frozen=True)
class Projection:
    plans: dict[str, Plan]
    total: int
    ground_wait: int
    rider_minutes: int

    def score(self):
        return -len(self.plans), self.total, self.ground_wait, self.rider_minutes


def rider_minutes(plan):
    return sum(s.end - s.start for s in plan.slots if s.resource.startswith("rider:"))


def _window(engine, pivot, policy):
    known = [record["order"] for record in engine.records.values()
             if record["status"] in ("waiting", "unready") and record["order"].id != pivot.id
             and record["order"].arrival <= engine.now]
    known.sort(key=engine._ready_rank)
    if not policy.diversify:
        return known[:policy.window]
    ready = [row for row in known if row.ready <= engine.now]
    soon = [row for row in known if engine.now < row.ready <= engine.now + REGRET_MINUTES]
    # Reserve at most two seats for orders already received and almost ready.
    seats = min(2, policy.window // 4, len(soon))
    return sorted(ready[:policy.window-seats] + soon[:max(seats, policy.window-len(ready))],
                  key=engine._ready_rank)


def _wait_event(engine, pivot, window):
    left = WAIT_BUDGET - engine.lookahead_wait_spent.get(pivot.id, 0)
    if left <= 0 or any(row.ready <= engine.now for row in window):
        return None
    times = {row.ready for row in window}
    times.update(engine.rider_start(rider, max(engine.now + 1, end)) for rider, (end, _) in engine.cal.tails.items()
                 if end > engine.now or engine.rider_start(rider, engine.now) > engine.now)
    for resource, slots in engine.cal.slots.items():
        if resource.startswith(("aircraft:", "pad:", "swap:", "charge:")):
            times.update(slot.end for slot in slots if slot.start <= engine.now < slot.end)
    times.update(end for limits in engine.cal.limits.values() for start, end in limits.values()
                 if start <= engine.now < end)
    return min((t for t in times if engine.now < t <= engine.now+left), default=None)


def _actions(engine, pivot, reference, window, policy, cancelled):
    choices = []
    cutoff = reference.eta + REGRET_MINUTES
    if reference.mode == "air":
        ground = engine._ground(pivot, engine.cal, engine.now, True)
        if ground and ground.eta <= cutoff:
            choices.append(Action(pivot, ground, "ground"))
    if not pivot.ground_only and policy.diversify:
        plans = []
        for route_id in pivot.candidate_route_ids or (pivot.route_id,):
            if cancelled():
                break
            plan = engine._air_route(pivot, engine.routes[route_id], engine.cal, engine.now, True, cutoff, cancelled)
            if plan and plan != reference:
                plans.append(plan)
        if plans:
            fastest = min(plans, key=lambda p: (p.eta, p.route))
            least_rider = min(plans, key=lambda p: (rider_minutes(p), p.eta, p.route))
            for plan in dict.fromkeys((fastest, least_rider)):
                choices.append(Action(pivot, plan, "air"))
    elif reference.mode == "ground" and not pivot.ground_only:
        air = engine._air(pivot, engine.cal, engine.now, True, cutoff=cutoff, cancelled=cancelled)
        if air:
            choices.append(Action(pivot, air, "air"))
    if policy.diversify and policy.peer_first and not pivot.ground_only and engine.lookahead_bypasses.get(pivot.id, 0) < 1:
        peer = next((row for row in window if row.ground_only and row.ready <= engine.now
                     and engine.lookahead_wait_until.get(row.id, 0) <= engine.now), None)
        if peer:
            plan = engine._ground(peer, engine.cal, engine.now, True)
            if plan:
                choices.append(Action(peer, plan, "ground_first"))
    if policy.event_wait and (event := _wait_event(engine, pivot, window)):
        choices.append(Action(pivot, None, "event_wait", event))
    return choices


def _project(engine, selected, orders, cancelled):
    calendar = engine.cal.fork()
    planned = {}
    if selected.plan:
        calendar.reserve(selected.plan)
        planned[selected.order.id] = selected.plan
    start = selected.until or engine.now
    for row in orders:
        if cancelled():
            return None
        if row.id not in planned:
            available = max(start, engine.lookahead_wait_until.get(row.id, 0))
            plan = engine._best(engine.opts(row, calendar, available, fastest=True, cancelled=cancelled))
            if plan:
                calendar.reserve(plan)
                planned[row.id] = plan
    measured = [(row, planned[row.id]) for row in orders if row.id in planned]
    return Projection(planned, sum(plan.eta-row.arrival for row, plan in measured),
                      sum(plan.first-row.ready for row, plan in measured if row.ground_only),
                      sum(rider_minutes(plan) for _, plan in measured))


def _summary(action, projection):
    return {"kind": action.kind, "order": action.order.id, "until": action.until,
            "route": action.plan.route if action.plan else None,
            "first_eta": action.plan.eta if action.plan else None,
            "completed": len(projection.plans), "total_minutes": projection.total,
            "ground_wait": projection.ground_wait, "rider_minutes": projection.rider_minutes}


def choose(engine, pivot, plan, cancelled):
    stats = engine.lookahead_stats
    reference = Action(pivot, plan, "dynamic")
    policy = engine.lookahead_policy
    window = _window(engine, pivot, policy)
    if not window:
        stats["dynamic_no_peers"] += 1
        return reference
    choices = _actions(engine, pivot, plan, window, policy, cancelled)
    if not choices:
        stats["dynamic_no_alternative"] += 1
        return reference
    stats["joint_decisions"] += 1
    orders = sorted([pivot, *window], key=engine._ready_rank)
    base = _project(engine, reference, orders, cancelled)
    if base is None:
        return reference
    best, selected = base, reference
    matched_ground = [row.id for row in orders if row.ground_only and row.id in base.plans]
    trace = getattr(engine, "lookahead_trace", None)
    candidates = [_summary(reference, base)] if trace is not None else None
    for action in choices:
        trial = _project(engine, action, orders, cancelled)
        if trial is None:
            return reference
        stats["projected_alternatives"] += 1
        feasible = base.plans.keys() <= trial.plans.keys()
        if feasible and policy.protect_ground:
            extra_wait = sum(trial.plans[key].first - base.plans[key].first for key in matched_ground)
            if extra_wait > 0:
                feasible = False
                stats["blocked_ground_wait"] += 1
        if candidates is not None:
            candidates.append({**_summary(action, trial), "eligible": feasible})
        if feasible and trial.score() < best.score():
            best, selected = trial, action
    stats["dynamic_selected" if selected is reference else selected.kind] += 1
    if selected is not reference:
        stats["projected_minutes_saved_on_changes"] += base.total-best.total
    if trace is not None and selected is not reference:
        trace.append({"time": engine.now, "pivot": pivot.id, "window": [row.id for row in orders],
                      "candidates": candidates, "selected": _summary(selected, best),
                      "dynamic_projected_eta": {key: plan.eta for key, plan in base.plans.items()},
                      "selected_projected_eta": {key: plan.eta for key, plan in best.plans.items()},
                      "selected_slots": [vars(s) for s in selected.plan.slots] if selected.plan else []})
    return selected
