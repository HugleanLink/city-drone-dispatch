"""Single recommendations and joint allocation against the same finite fleet."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from itertools import islice, product
from math import ceil, prod

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize

from .deployment import compromise, value
from .model import match_order, minutes
from .sim import Plan, Slot


DAY = 1440
SHIFT = 480
MAX_ORDERS = 10


def absolute_minute(timestamp):
    stamp = datetime.fromisoformat(timestamp)
    return stamp.toordinal() * DAY + 60 * stamp.hour + stamp.minute


def staff_estimate(profile):
    frame = profile.rows
    valid = frame.reference_minutes.gt(0)
    count = frame.loc[valid, "count"]
    work = (count * frame.loc[valid, "reference_minutes"]).groupby(frame.loc[valid, "minute"] // 60).sum()
    hourly = (work.reindex(range(24), fill_value=0) * 1.3 / (60 * .8 * frame.group.nunique()))
    return tuple(ceil(max(hourly[i:i + 8])) for i in (0, 8, 16))


@dataclass(frozen=True)
class Schedule:
    riders: dict = field(default_factory=dict)
    aircraft: dict = field(default_factory=dict)

    def reserve(self, reservation):
        riders = dict(self.riders)
        aircraft = dict(self.aircraft)
        for identifier, end, location in reservation["rider_ends"]:
            if identifier not in riders or end > riders[identifier][0]:
                riders[identifier] = (end, tuple(location))
        for slot in reservation["slots"]:
            name = slot["resource"]
            if name.startswith("aircraft:"):
                aircraft[name] = max(slot["end"], aircraft.get(name, 0))
        return Schedule(riders, aircraft)


def saved_schedule(records):
    state = Schedule()
    for row in records:
        if row.reservation:
            state = state.reserve(row.reservation)
    return state


def grounds_used(plan):
    return sum(slot.end - slot.start for slot in plan.slots if slot.resource.startswith("rider:"))


def operating_cost(plan, routes):
    return .20 * grounds_used(plan) + (0 if plan.route is None else
                                     2 * minutes(routes[plan.route].km, 65) * .15 + .50 + .50)


@dataclass(frozen=True)
class Job:
    order: object
    ordered_at: str
    arrival: int
    ready: int


class Dispatcher:
    def __init__(self, network, speed, riders, aircraft):
        self.network, self.speed = network, speed
        self.routes = {route.id: route for route in network.routes}
        self.hub_aircraft = dict(aircraft)
        starts = network.rider_starts or (network.origin,)
        self.roster = {f"rider:{shift}:{i+1}": (shift, starts[i % len(starts)])
                       for shift, count in enumerate(riders) for i in range(count)}
        self.initial = {}
        for name, (shift, point) in self.roster.items():
            self.initial.setdefault((shift, 0, point), []).append(name)
        self._leg_times = lru_cache(maxsize=32_768)(self._leg_times)

    def job(self, order, stamp):
        order = match_order(self.network, order)
        arrival = absolute_minute(stamp)
        return Job(order, stamp, arrival, arrival + order.ready - order.arrival)

    def _groups(self, state):
        groups = {key: list(islice((name for name in names if name not in state.riders), 2))
                  for key, names in self.initial.items()}
        for name, (free, point) in state.riders.items():
            if name not in self.roster:
                continue
            shift, _ = self.roster[name]
            group = groups.setdefault((shift, free, point), [])
            if len(group) < 2:
                group.append(name)
        return [(name, shift, free, point) for (shift, free, point), group in groups.items() for name in group]

    def _leg(self, crew, origin, destination, available, now, *, terminal=0, just_in_time=False):
        name, shift, free, position = crew
        times = self._leg_times(shift, free, position, origin, destination, available, now, terminal, just_in_time)
        return (name, *times) if times else None

    def _leg_times(self, shift, free, position, origin, destination, available, now, terminal, just_in_time):
        earliest = max(now, free)
        for day in range(now // DAY, now // DAY + 2):
            opening = day * DAY + shift * SHIFT
            closing = opening + SHIFT
            start = max(opening, earliest)
            if start >= closing:
                continue
            empty = self.speed.travel(position, origin, start)
            fetch = max(start + empty, available) + 1
            end = fetch + self.speed.travel(origin, destination, fetch) + terminal
            if just_in_time and start + empty < available:
                later = available - empty
                later_fetch = max(later + self.speed.travel(position, origin, later), available) + 1
                later_end = later_fetch + self.speed.travel(origin, destination, later_fetch) + terminal
                if later_end <= end:
                    start, fetch, end = later, later_fetch, later_end
            if end <= closing:
                return start, fetch, end
        return None

    def ground(self, job, state, now, groups=None):
        order = job.order
        options = (self._leg(crew, order.pickup, order.dropoff, job.ready, now)
                   for crew in (self._groups(state) if groups is None else groups))
        best = min((leg for leg in options if leg), key=lambda leg: (leg[3], leg[1], leg[0]), default=None)
        if not best:
            return None
        rider, start, fetch, eta = best
        return Plan("ground", start, eta, fetch, (Slot(rider, start, eta, "地面直送", order.id),),
                    ((rider, eta, order.dropoff),))

    def air(self, job, route, state, now, groups=None):
        if route.km * 2 > 20:
            return None
        planes = self.hub_aircraft.get(route.hub, 0)
        if not planes:
            return None
        ready, plane = min((state.aircraft.get(name, 0), name)
                           for i in range(planes) if (name := f"aircraft:{route.hub}:{i+1}"))
        groups = self._groups(state) if groups is None else groups
        order = job.order
        flight = minutes(route.km, 65)
        best, score = None, None
        for crew in groups:
            first = self._leg(crew, order.pickup, route.pickup, job.ready, now, terminal=1)
            if not first:
                continue
            rider, start, fetch, hub = first
            mount = max(hub, ready)
            launch = mount + 2
            parcel = launch + flight + 2
            if score is not None and parcel + 1 > score[0]:
                continue
            for other in groups:
                person = (rider, crew[1], hub, route.pickup) if other[0] == rider else other
                last = self._leg(person, route.dropoff, order.dropoff, parcel, now, just_in_time=True)
                if not last:
                    continue
                final, begin, transfer, eta = last
                key = (eta, (hub - start) + (eta - begin), start, rider, final)
                if score is not None and key >= score:
                    continue
                slots = (Slot(rider, start, hub, "首段接驳", order.id),
                         Slot(plane, mount, parcel + flight + 3, "航空往返及周转", order.id),
                         Slot(final, begin, eta, "末段接驳", order.id))
                best = Plan("air", min(start, begin), eta, fetch, slots,
                            ((rider, hub, route.pickup), (final, eta, order.dropoff)), route.id,
                            (hub, launch, parcel, transfer))
                score = key
        return best

    def options(self, job, state, now):
        groups = self._groups(state)
        ground = self.ground(job, state, now, groups)
        options = (self.air(job, self.routes[key], state, now, groups) for key in job.order.candidate_route_ids)
        air = min((plan for plan in options if plan), key=lambda p: (p.eta, grounds_used(p), p.route), default=None)
        return ground, air

    def immediate(self, job, state, now, threshold=0):
        ground, air = self.options(job, state, now)
        if ground and air:
            chosen = air if ground.eta - air.eta > threshold else ground
        else:
            chosen = ground or air
        return ground, air, chosen


class RouteMutation(Mutation):
    def _do(self, problem, X, *args, random_state=None, **kwargs):
        result = X.copy()
        for k, limit in enumerate(problem.maximum):
            if limit:
                for i in np.flatnonzero(random_state.random(len(X)) < 1 / len(problem.maximum)):
                    offset = random_state.integers(1, limit + 1)
                    result[i, k] = (result[i, k] + offset) % (limit + 1)
        return result


class AllocationProblem(Problem):
    def __init__(self, dispatcher, jobs, state, now):
        self.dispatcher, self.jobs, self.now = dispatcher, jobs, now
        self.maximum = tuple(len(job.order.candidate_route_ids) for job in jobs)
        self.weights = [1 + value(job.order, *(dispatcher.network.diagnostics or {}).get("price_cutoffs", (None, None)))
                        for job in jobs]
        self.prefixes = {(): (state, ())}
        super().__init__(n_var=len(jobs), n_obj=3, n_ieq_constr=1, xl=0, xu=np.array(self.maximum), vtype=int)

    def decode(self, genes):
        key = tuple(map(int, genes))
        if key not in self.prefixes:
            state, plans = self.decode(key[:-1])
            job = self.jobs[len(key) - 1]
            route = job.order.candidate_route_ids[key[-1] - 1] if key[-1] else None
            plan = (self.dispatcher.air(job, self.dispatcher.routes[route], state, self.now)
                    if route else self.dispatcher.ground(job, state, self.now))
            self.prefixes[key] = (state.reserve(asdict(plan)) if plan else state, plans + (plan,))
        return self.prefixes[key]

    def _evaluate(self, genes, out, *args, **kwargs):
        objectives, missing = [], []
        for row in genes:
            _, plans = self.decode(row)
            completed = [(plan, job, weight) for plan, job, weight in zip(plans, self.jobs, self.weights) if plan]
            objectives.append((sum(weight * (plan.eta - job.arrival) for plan, job, weight in completed),
                               sum(grounds_used(plan) for plan, _, _ in completed),
                               sum(operating_cost(plan, self.dispatcher.routes) for plan, _, _ in completed)))
            missing.append([len(self.jobs) - len(completed)])
        out["F"], out["G"] = np.asarray(objectives), np.asarray(missing)


def joint(dispatcher, jobs, state, now, threshold=0, population=100, generations=200):
    if not 2 <= len(jobs) <= MAX_ORDERS:
        raise ValueError(f"联合调度一次处理 2 至 {MAX_ORDERS} 笔订单")
    if len({job.order.id for job in jobs}) != len(jobs) or any(job.arrival > now for job in jobs):
        raise ValueError("联合调度只接受不重复、已到达的订单")
    jobs = tuple(sorted(jobs, key=lambda job: job.arrival))
    problem = AllocationProblem(dispatcher, jobs, state, now)
    reference, plans = [], []
    trial = state
    for job in jobs:
        _, _, plan = dispatcher.immediate(job, trial, now, threshold)
        plans.append(plan)
        route = plan.route if plan else None
        reference.append(job.order.candidate_route_ids.index(route) + 1 if route else 0)
        if plan:
            trial = trial.reserve(asdict(plan))

    domains = [range(v + 1) for v in problem.maximum]
    rng = np.random.default_rng(42)
    space = prod(map(len, domains))
    if space <= population:
        samples = list(product(*domains))
    else:
        samples = [tuple(0 for _ in jobs), tuple(reference)]
        while len(samples) < population:
            row = tuple(rng.integers(0, high + 1) for high in problem.maximum)
            if row not in samples:
                samples.append(row)
    result = minimize(problem, NSGA2(pop_size=len(samples), sampling=np.asarray(samples),
                                    crossover=UniformCrossover(prob=.9), mutation=RouteMutation(),
                                    eliminate_duplicates=True),
                      ("n_gen", generations if space > population else 1), seed=42)
    if result.X is None:
        return jobs, tuple(plans), {"fallback": True, "pareto": 0, "objectives": None}
    objectives = np.atleast_2d(result.F)
    selected = compromise(objectives)
    _, chosen = problem.decode(np.atleast_2d(result.X)[selected])
    return jobs, chosen, {"fallback": False, "pareto": len(objectives),
                          "objectives": objectives[selected].tolist()}
