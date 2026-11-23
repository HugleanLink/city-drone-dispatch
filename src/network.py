from dataclasses import asdict, replace
from math import ceil, floor
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from .model import FlightRules, Network, Route, match_order


def _points(network, endpoints, prefix, limit, eps, minimum):
    xy = np.asarray([network.xy(point) for point in endpoints])
    unique, first, inverse, counts = np.unique(
        xy, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    # Preserve first occurrence, which DBSCAN uses for ambiguous border points.
    order = np.argsort(first)
    samples = unique[order]
    weights = counts[order]
    if len(samples) <= 4_000:
        cells, cell_weights, positions = samples, weights, np.arange(len(samples))
    else:
        # Bound neighborhood memory for dense, non-gridded input.
        cells, positions = np.unique(np.floor(samples * 2 + .5) / 2, axis=0, return_inverse=True)
        cell_weights = np.bincount(positions, weights=weights).astype(int)
    if eps is None:
        neighbors = NearestNeighbors(n_neighbors=min(5, len(cells))).fit(cells)
        distances, indices = neighbors.kneighbors(cells)
        fifth = (cell_weights[indices].cumsum(axis=1) < 5).sum(axis=1).clip(max=distances.shape[1]-1)
        eps = max(.1, float(np.percentile(np.repeat(distances[np.arange(len(cells)), fifth], cell_weights), 95)))
    minimum = minimum if minimum is not None else max(3, int(len(endpoints) * .005))
    labels = DBSCAN(eps=eps, min_samples=minimum).fit_predict(cells, sample_weight=cell_weights)
    ordered_labels = labels[positions]
    unique_labels = np.empty(len(unique), dtype=int)
    unique_labels[order] = ordered_labels
    high_frequency = sorted(zip(map(tuple, unique), counts), key=lambda item: (-item[1], *item[0]))
    clusters = sorted(set(unique_labels) - {-1},
                      key=lambda label: (-int(counts[unique_labels == label].sum()), label))
    representatives = []
    for label in clusters:
        group = unique_labels == label
        cluster = unique[group]
        center = np.average(cluster, axis=0, weights=counts[group])
        representatives.append(min(map(tuple, cluster), key=lambda p: (np.linalg.norm(p - center), *p)))
    selected = list(dict.fromkeys(representatives + [point for point, _ in high_frequency]))[:limit]
    frequency = {point: int(n) for point, n in high_frequency}
    return [(f"{prefix}{i+1:02}", network.latlon(*p), frequency[p])
            for i, p in enumerate(sorted(selected))], unique_labels[inverse], eps, minimum


def build(orders, origin=None, *, max_hubs=3, max_spokes=4, max_routes=6, extent=None,
          rules=FlightRules(), eps=.55, minimum=10):
    if not orders:
        raise ValueError("请先生成规划订单")
    points = [point for order in orders for point in (order.pickup,order.dropoff)]
    points.extend(extent or ())
    if origin is None:
        origin = tuple((min(axis)+max(axis))/2 for axis in zip(*points))
    base = Network(origin,(0,0,0,0),(), rules=rules)
    xs, ys = zip(*(base.xy(p) for p in points))
    bounds = (floor(min(xs)*2)/2-.5, ceil(max(xs)*2)/2+.5,
              floor(min(ys)*2)/2-.5, ceil(max(ys)*2)/2+.5)
    base = replace(base, bounds=bounds)
    starts=[]
    pickup=np.array([base.xy(o.pickup) for o in orders])
    for east,north in ((False,False),(False,True),(True,False),(True,True)):
        local=pickup[((pickup[:,0]>=0)==east)&((pickup[:,1]>=0)==north)]
        x,y=local.mean(axis=0) if len(local) else (bounds[1 if east else 0]/2,bounds[3 if north else 2]/2)
        starts.append(base.latlon(ceil(x*2-.5)/2,ceil(y*2-.5)/2))
    base=replace(base,rider_starts=tuple(starts))
    rejected = Counter()
    potential = []
    for order in orders:
        reason = base.rejection(order)
        if reason:
            rejected[reason] += 1
        else:
            potential.append(order)
    stats = {"planning": len(orders), "potential": len(potential), "covered": 0, "routes": 0,
             "screened_out": dict(rejected)}
    if not potential:
        return replace(base, diagnostics=stats)
    hubs, origins, pickup_eps, pickup_min = _points(
        base, [o.pickup for o in potential], "H", max(38 if rules.report else 20, max_hubs), eps, minimum)
    spokes, destinations, dropoff_eps, dropoff_min = _points(
        base, [o.dropoff for o in potential], "S", max(23 if rules.report else 20, max_spokes),
        pickup_eps if rules.report else eps, pickup_min if rules.report else minimum)
    pairs = {}
    pair_index = np.fromiter((pairs.setdefault((o.pickup, o.dropoff), len(pairs)) for o in potential),
                             dtype=np.intp, count=len(potential))
    totals = np.bincount(pair_index)
    values = np.array([_value(o) for o in potential]) if rules.report else np.ones(len(potential))
    value_totals = np.bincount(pair_index, weights=values)
    reference = np.array([np.nan if o.reference_minutes is None else o.reference_minutes for o in potential])
    observed = np.isfinite(reference)
    long = observed & (reference > np.percentile(reference[observed], 90)) if observed.any() else observed
    observed_counts = np.bincount(pair_index, weights=observed)
    long_counts = np.bincount(pair_index, weights=long)
    direct = np.ceil(np.array([base.road(a, b) for a, b in pairs]) * 60 / 18 - 1e-10).astype(np.int32)
    first_legs = _legs(base, hubs, [a for a, _ in pairs])
    last_legs = _legs(base, spokes, [b for _, b in pairs])
    legs = []
    for hid, hub, first, first_time in first_legs:
        for sid, spoke, last, last_time in last_legs:
            km = base.distance(hub, spoke)
            if not rules.leg_min <= km <= rules.leg_max:
                continue
            mask = first & last
            if mask.any():
                route = Route(f"{hid}-{sid}", hid, sid, hub, spoke, km, int(totals[mask].sum()),
                              _site(hub), _site(spoke))
                legs.append((route, mask, first_time, last_time))
    covered = np.zeros(len(pairs), dtype=bool)
    used_h, used_s, chosen = set(), set(), []
    observed_by_hub = {hid: observed_counts[mask].sum() for hid, _, mask, _ in first_legs}
    bonuses = {hid: 1 + .3 * long_counts[mask].sum() / observed_by_hub[hid] if observed_by_hub[hid] else 1.
               for hid, _, mask, _ in first_legs}
    selected_hubs = _report_hubs(first_legs, bonuses, value_totals, max_hubs) if rules.report else set()
    remaining = [item for item in legs if not rules.report or item[0].hub in selected_hubs]
    while remaining and (max_routes is None or len(chosen) < max_routes):
        def rank(item):
            route, mask, first_time, last_time = item
            added = mask & ~covered
            score = value_totals[added].sum() * (bonuses[route.hub] if rules.report else 1.)
            saving = ((direct[added] - first_time[added] - last_time[added]) * totals[added]).sum()
            return (-score, -route.coverage if rules.report else -int(saving), route.km, route.id)

        legal = [(index, item) for index, item in enumerate(remaining)
                 if (item[0].hub in used_h or len(used_h) < max_hubs)
                 and (item[0].spoke in used_s or len(used_s) < max_spokes)
                 and (not rules.report or item[0].hub not in used_h or item[0].spoke not in used_s)]
        selected = min(legal, key=lambda pair: rank(pair[1]), default=None)
        if selected is None:
            break
        index, (route, mask, _, _) = selected
        if not rules.report and not (mask & ~covered).any():
            break
        remaining.pop(index)
        covered |= mask
        chosen.append(route)
        used_h.add(route.hub)
        used_s.add(route.spoke)
    if rules.report:
        for route, mask, _, _ in sorted(remaining, key=lambda item: (-item[0].coverage, item[0].id)):
            if max_routes is not None and len(chosen) >= max_routes:
                break
            if route.hub in used_h and route.spoke in used_s:
                covered |= mask
                chosen.append(route)
    active_hubs = selected_hubs if rules.report else used_h
    candidates = tuple(dict(id=i, location=point, count=n, role=role, active=i in active)
                       for role, group, active in (("Hub",hubs,active_hubs),("Spoke",spokes,used_s))
                       for i,point,n in group)
    base=replace(base,routes=tuple(chosen),candidates=candidates)
    stats.update({"covered":int(totals[covered].sum()),"routes":len(chosen),
             "pickup_eps_km":pickup_eps,"dropoff_eps_km":dropoff_eps,
             "pickup_min_samples":pickup_min,"dropoff_min_samples":dropoff_min,
             "pickup_clusters":len(set(origins)-{-1}),"dropoff_clusters":len(set(destinations)-{-1}),
             "pickup_noise":int((origins==-1).sum()),"dropoff_noise":int((destinations==-1).sum()),
             "cluster_flows":[{"pickup":int(a),"dropoff":int(b),"orders":int(n)}
                              for (a,b),n in Counter(zip(origins,destinations)).most_common()]})
    best_eta = np.full(len(pairs), np.inf)
    winners = np.full(len(pairs), -1, dtype=int)
    chosen_by_id = {route.id: index for index, route in enumerate(chosen)}
    for route, mask, first_time, last_time in sorted(legs, key=lambda item: item[0].id):
        if route.id in chosen_by_id:
            eta = first_time + last_time + np.ceil(route.km * 60 / 40 - 1e-10) + 6
            better = mask & (eta < best_eta)
            best_eta[better] = eta[better]
            winners[better] = chosen_by_id[route.id]
    if covered.any():
        connections = []
        for (pickup, dropoff), winner in zip(pairs, winners):
            if winner >= 0:
                route = chosen[winner]
                connections.append((base.road(pickup, route.pickup), base.road(route.dropoff, dropoff)))
        stats["first_p90_km"], stats["last_p90_km"] = map(
            float, np.percentile(np.repeat(connections, totals[covered], axis=0), 90, axis=0))
        stats["nominal_air_faster"] = int(totals[best_eta < direct + 2].sum())
    if rules.report:
        hub_locations = [(name, site) for name, site, _ in hubs if name in active_hubs]
        stats["hub_corridors"] = []
        for index, (a, start) in enumerate(hub_locations):
            for b, end in hub_locations[index+1:]:
                km = base.distance(start, end)
                if 0 < km <= rules.order_max * 1.5 and km * 60 / 65 + 15 <= 45:
                    stats["hub_corridors"].append(dict(id=f"{a}-{b}", start=start, end=end, km=km))
    return replace(base,diagnostics=stats)


def _report_hubs(first_legs, bonuses, values, limit):
    covered = np.zeros(len(values), dtype=bool)
    selected = set()
    remaining = {name: mask for name, _, mask, _ in first_legs}
    while remaining and len(selected) < limit:
        name = max(remaining, key=lambda hub: values[covered | remaining[hub]].sum() * bonuses[hub])
        covered |= remaining.pop(name)
        selected.add(name)
    return selected


def _value(order):
    if order.business == "health":
        return 3.
    if order.price is not None and order.price >= 56:
        return 2.
    if order.price is not None and order.price >= 30.8:
        return 1.
    return .5


def _legs(base, candidates, endpoints):
    positions = list(dict.fromkeys(endpoints))
    indices = {point: i for i, point in enumerate(positions)}
    inverse = np.asarray([indices[point] for point in endpoints])
    output = []
    for site_id, site, _ in candidates:
        access = np.asarray([base.access(point, site) for point in positions])[inverse]
        road = np.asarray([base.road(point, site) for point in positions])[inverse]
        output.append((site_id, site, access <= base.rules.access_km,
                       np.ceil(road * 60 / 18 - 1e-10).astype(np.int32)))
    return output


def _site(point):
    return f"{point[0]:.8f},{point[1]:.8f}"


def covered_rows(network, orders):
    return [asdict(match_order(network,o)) for o in orders]
