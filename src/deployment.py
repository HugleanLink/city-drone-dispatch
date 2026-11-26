"""Data-driven Hub, Spoke and corridor deployment."""

from dataclasses import replace
from itertools import combinations
from math import ceil, floor

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from .model import FlightRules, Network, Route, haversine
from .speed import distance_screen, distances_between


def order_values(orders):
    prices = [o.price for o in orders if o.business != "pinhaofan" and o.price is not None]
    median, upper = map(float, np.quantile(prices, [.5, .75])) if prices else (None, None)
    return [value(o, median, upper) for o in orders], [median, upper]


def value(order, median, upper):
    if order.business == "health":
        return 3.
    if order.business == "pinhaofan":
        return 0.
    if order.price is None or median is None:
        return 1.
    if order.price >= upper:
        return 2.
    return 1. if order.price >= median else .5


def compromise(objectives):
    matrix = np.asarray(objectives)
    low = matrix.min(axis=0)
    span = matrix.max(axis=0) - low
    distance = ((matrix - low) / np.where(span > 0, span, 1)).max(axis=1)
    return min(range(len(matrix)), key=lambda i: (distance[i], *matrix[i]))


def _grid(pickup, dropoff):
    step = .005
    keys, inverse = np.unique(np.floor(np.concatenate([pickup, dropoff]) / step).astype(int),
                              axis=0, return_inverse=True)
    p = np.bincount(inverse[:len(pickup)], minlength=len(keys))
    d = np.bincount(inverse[len(pickup):], minlength=len(keys))
    features = np.column_stack([p + d, p / (d + 1)])
    if len(np.unique(features, axis=0)) < 4:
        return (), ()
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(features)
    flow, ratio = np.median(model.cluster_centers_, axis=0)
    hubs, spokes = [], []
    for key, label in zip(keys, model.labels_):
        f, r = model.cluster_centers_[label]
        if f > flow:
            (hubs if r > ratio else spokes).append(tuple((key + .5) * step))
    return hubs, spokes


def _clusters(base, pickup, dropoff):
    def points(coords):
        unique, counts = np.unique(coords, axis=0, return_counts=True)
        xy = np.asarray([base.xy(p) for p in unique]) * 1000
        return unique, counts, xy

    p, counts, xy = points(pickup)
    neighbors = NearestNeighbors(n_neighbors=min(5, len(p))).fit(xy)
    distance, index = neighbors.kneighbors(xy)
    needed = min(5, len(pickup))
    kth = (counts[index].cumsum(axis=1) >= needed).argmax(axis=1)
    eps = max(100., np.quantile(np.repeat(distance[np.arange(len(p)), kth], counts), .95))
    minimum = max(3, int(.005 * len(pickup)))

    def centers(unique, frequency, coords):
        labels = DBSCAN(eps=float(eps), min_samples=minimum).fit_predict(coords, sample_weight=frequency)
        return [tuple(np.average(unique[labels == cluster], axis=0, weights=frequency[labels == cluster]))
                for cluster in sorted(set(labels) - {-1})]

    return centers(p, counts, xy), centers(*points(dropoff))


def _choose_hubs(pickup, candidates, weights, duration, radius, limit):
    points, inverse = np.unique(pickup, axis=0, return_inverse=True)
    cover = distances_between(np.asarray(candidates)[:, None], points[None, :]) <= radius
    valid = np.isfinite(duration)
    slow = duration > np.quantile(duration[valid], .9) if valid.any() else valid
    counts = np.bincount(inverse)
    bonus = 1 + .3 * (cover @ np.bincount(inverse, weights=slow) / np.maximum(cover @ counts, 1))
    value_at_pickup = np.bincount(inverse, weights=weights)
    remaining = list(range(len(candidates)))
    selected = []
    union = np.zeros(len(points), dtype=bool)
    for _ in range(min(limit, len(candidates))):
        scores = ((cover[remaining] | union) @ value_at_pickup) * bonus[remaining]
        best = remaining.pop(int(scores.argmax()))
        union |= cover[best]
        selected.append(candidates[best])
    return selected, union[inverse]


def _coverage(sites, endpoints, radius):
    points, inverse = np.unique(endpoints, axis=0, return_inverse=True)
    cover = distances_between(np.asarray(sites).reshape(-1, 1, 2), points[None, :]) <= radius
    return cover[:, inverse]


class DeploymentProblem(Problem):
    def __init__(self, hubs, parents, corridors, cover, weights, estimated):
        self.hubs = hubs
        self.parents = np.asarray(parents, dtype=int)
        self.corridors = np.asarray(corridors, dtype=int).reshape(-1, 2)
        packed = np.packbits(cover.T, axis=1)
        patterns, inverse = np.unique(packed, axis=0, return_inverse=True)
        self.cover = np.unpackbits(patterns, axis=1, count=len(parents)).T.astype(float)
        self.counts = np.bincount(inverse)
        self.weights = np.bincount(inverse, weights=weights)
        self.estimated = estimated
        super().__init__(n_var=hubs + len(parents) + len(corridors), n_obj=3, xl=0, xu=1)

    def decode(self, population):
        opened = np.atleast_2d(population) > .5
        h = opened[:, :self.hubs]
        s = opened[:, self.hubs:self.hubs + len(self.parents)] & h[:, self.parents]
        e = opened[:, self.hubs + len(self.parents):] & h[:, self.corridors].all(axis=2)
        return h, s, e

    def _evaluate(self, population, out, *args, **kwargs):
        h, s, _ = self.decode(population)
        covered = (s @ self.cover) > 0
        counts = covered @ self.counts
        values = covered @ self.weights
        # Five-year facility costs from their investment and operating components.
        hub = 250_000 + 5 * (12 * 180 * 50 + 24_000 + 15_000 + 120_000 * .5)
        spoke = 60_000 + 5 * (12 * 120 * 20 + 8_000 + 5_000 + 96_000 * .3)
        cost = h.sum(axis=1) * hub + s.sum(axis=1) * spoke + counts * (2.8 - 6.5)
        out["F"] = np.column_stack([cost, 1 - values / (self.weights.sum() or 1),
                                    1 - counts / max(self.estimated, 1)])


def deploy(orders, origin, extent, *, max_hubs=15, weight=2., radius=3.,
           population=100, generations=200):
    base = Network(origin, (0, 0, 0, 0), ())
    low, high = (base.xy(point) for point in extent)
    bounds = (floor(low[0] * 2) / 2 - .5, ceil(high[0] * 2) / 2 + .5,
              floor(low[1] * 2) / 2 - .5, ceil(high[1] * 2) / 2 + .5)
    pickup = np.asarray([o.pickup for o in orders], dtype=float).reshape(-1, 2)
    dropoff = np.asarray([o.dropoff for o in orders], dtype=float).reshape(-1, 2)
    km = distances_between(pickup, dropoff)
    duration = np.asarray([o.reference_minutes for o in orders], dtype=float)
    clean = (km >= .1) & (km <= 20) & (duration >= 0) & (duration <= 180)
    lo, hi, ratio, bic, fallback = distance_screen(km[clean])
    rules = FlightRules(weight=weight, order_min=lo, order_max=hi, access_km=radius,
                        leg_min=.000001, leg_max=10., report=True)
    base = replace(base, bounds=bounds, rules=rules)
    suitable = clean & (km >= lo) & (km <= hi)
    values, price_cutoffs = order_values([o for o, ok in zip(orders, suitable) if ok])
    weights = np.asarray(values)
    p, d = pickup[suitable], dropoff[suitable]
    info = {"planner": "GMM / K-means / DBSCAN / NSGA-II", "screened_orders": len(p),
            "distance_window": [lo, hi], "candidate_fraction": ratio, "distance_bic": bic,
            "distance_fallback": fallback, "price_cutoffs": price_cutoffs, "hub_corridors": []}
    if not len(p):
        return replace(base, diagnostics=info)
    hubs, spokes = _grid(p, d)
    info["grid_candidates"] = [len(hubs), len(spokes)]
    if not hubs or not spokes:
        hubs, spokes = _clusters(base, p, d)
        info["candidate_method"] = "DBSCAN"
    else:
        info["candidate_method"] = "网格 K-means"
    if not hubs:
        return replace(base, diagnostics=info)
    hubs, start_cover = _choose_hubs(p, hubs, weights, duration[suitable], radius, max_hubs)
    end_cover = _coverage(hubs, d, radius).any(axis=0)
    combined = start_cover & end_cover
    info["candidate_order_coverage"] = float(combined.mean())
    info["candidate_value_coverage"] = float(weights[combined].sum() / (weights.sum() or 1))
    distance = distances_between(np.asarray(spokes).reshape(-1, 1, 2), np.asarray(hubs)[None, :])
    parents = distance.argmin(axis=1)
    pairs = [(a, b) for a, b in combinations(range(len(hubs)), 2)
             if (length := haversine(hubs[a], hubs[b])) <= 1.5 * hi and 60 * length / 65 + 15 <= 45]
    cover = _coverage(spokes, p, radius) & _coverage(spokes, d, radius)
    problem = DeploymentProblem(len(hubs), parents, pairs, cover, weights, int(len(p) / ratio))
    with threadpool_limits(limits=1):
        result = minimize(problem, NSGA2(pop_size=population, crossover=SBX(eta=15, prob=.9),
                                        mutation=PM(eta=20, prob=.1), eliminate_duplicates=True),
                          ("n_gen", generations), seed=42)
    objectives = np.atleast_2d(result.F)
    selected = compromise(objectives)
    active_h, active_s, active_e = (arr[0] for arr in problem.decode(np.atleast_2d(result.X)[selected]))
    info["pareto_solutions"] = len(objectives)
    info["deployment_objectives"] = objectives[selected].tolist()
    info["alternatives"] = {
        label: objectives[index].tolist()
        for label, index in (("A", objectives[:, 0].argmin()), ("B", objectives[:, 1].argmin()), ("C", selected))
    }
    sites = [dict(id=f"H{i+1}", role="Hub", location=point, active=bool(active_h[i]))
             for i, point in enumerate(hubs)]
    sites.extend(dict(id=f"S{i+1}", role="Spoke", location=point, active=bool(active_s[i]),
                      assigned_hub=f"H{parents[i]+1}", isolated=bool(distance[i, parents[i]] > radius))
                 for i, point in enumerate(spokes))
    enabled = [pair for pair, active in zip(pairs, active_e) if active]
    info["hub_corridors"] = [dict(id=f"H{a+1}-H{b+1}", start=hubs[a], end=hubs[b],
                                  km=haversine(hubs[a], hubs[b])) for a, b in enabled]
    routes = runtime_routes(hubs, spokes, parents, active_h, active_s, enabled)
    selection = np.random.default_rng(42).choice(len(orders), min(4, len(orders)), replace=False)
    initial = tuple(orders[i].pickup for i in selection)
    return replace(base, candidates=tuple(sites), routes=tuple(routes), diagnostics=info,
                   rider_starts=initial if len(initial) == 4 else ())


def runtime_routes(hubs, spokes, parents, active_h, active_s, corridors):
    paths = {(i, i): (0., ()) for i, active in enumerate(active_h) if active}
    for a, b in corridors:
        length = haversine(hubs[a], hubs[b])
        paths[a, b], paths[b, a] = (length, (b,)), (length, (a,))
    for k in range(len(hubs)):
        for a in range(len(hubs)):
            for b in range(len(hubs)):
                if (a, k) in paths and (k, b) in paths:
                    first, second = paths[a, k], paths[k, b]
                    length = first[0] + second[0]
                    if length < paths.get((a, b), (float("inf"), ()))[0]:
                        paths[a, b] = (length, first[1] + second[1])
    for i, spoke in enumerate(spokes):
        if not active_s[i]:
            continue
        parent = parents[i]
        term = haversine(hubs[parent], spoke)
        for h in range(len(hubs)):
            if (h, parent) not in paths:
                continue
            distance, stops = paths[h, parent]
            km = distance + term
            if .000001 <= km <= 10:
                hub, end = f"H{h+1}", f"S{i+1}"
                yield Route(f"{hub}-{end}", hub, end, hubs[h], spoke, km, 0,
                            via=tuple(hubs[j] for j in stops))
