from dataclasses import asdict, replace
from math import ceil, floor
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from .model import Network, Route, minutes, match_order


def _points(network, endpoints, prefix):
    xy = np.array([network.xy(point) for point in endpoints])
    labels = DBSCAN(eps=.55, min_samples=10).fit_predict(xy)
    unique, counts = np.unique(xy, axis=0, return_counts=True)
    high_frequency = sorted(zip(unique, counts), key=lambda x:(-x[1], *x[0]))
    representatives = []
    for label in sorted(set(labels)-{-1}, key=lambda label:(-int((labels==label).sum()),label)):
        cluster = xy[labels==label]
        center = cluster.mean(axis=0)
        representatives.append(min(map(tuple, cluster), key=lambda p:(np.linalg.norm(np.array(p)-center), *p)))
    selected = list(dict.fromkeys(representatives+[tuple(p) for p,_ in high_frequency]))[:20]
    frequency = {tuple(p):int(n) for p,n in high_frequency}
    return [(f"{prefix}{i+1:02}", network.latlon(*p), frequency[p]) for i,p in enumerate(sorted(selected))], labels


def build(orders, origin=None, *, max_hubs=3, max_spokes=4, max_routes=6, extent=None):
    if not orders:
        raise ValueError("请先生成规划订单")
    points = [point for order in orders for point in (order.pickup,order.dropoff)]
    points.extend(extent or ())
    if origin is None:
        origin = tuple((min(axis)+max(axis))/2 for axis in zip(*points))
    base = Network(origin,(0,0,0,0),())
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
    potential = [o for o in orders if o.kg<=2 and .5<=base.straight(o.pickup,o.dropoff)<=5]
    if not potential:
        return replace(base, diagnostics={"planning":len(orders),"potential":0,"covered":0,"routes":0})
    hubs, origins = _points(base,[o.pickup for o in potential],"H")
    spokes, destinations = _points(base,[o.dropoff for o in potential],"S")
    direct = np.array([minutes(base.road(o.pickup,o.dropoff),18)+2 for o in potential])
    legs = []
    for hid, hub, _ in hubs:
        first = np.array([base.road(o.pickup,hub) for o in potential])
        for sid, spoke, _ in spokes:
            km = base.straight(hub,spoke)
            if not .5 <= km <= 4:
                continue
            last = np.array([base.road(spoke,o.dropoff) for o in potential])
            mask = (first<=2.5)&(last<=2.5)
            if mask.any():
                saving = direct-np.ceil(first*60/18-1e-10)-np.ceil(last*60/18-1e-10)-4
                route = Route(f"{hid}-{sid}", hid,sid,hub,spoke,km,int(mask.sum()),
                              _site(hub),_site(spoke))
                legs.append((route,mask,saving))
    covered = np.zeros(len(potential), dtype=bool)
    used_h, used_s, chosen = set(), set(), []
    while len(chosen)<max_routes:
        legal = [(route,mask,saving) for route,mask,saving in legs
                 if (route.hub in used_h or len(used_h)<max_hubs)
                 and (route.spoke in used_s or len(used_s)<max_spokes)
                 and (mask & ~covered).any()]
        if not legal:
            break
        def rank(item):
            route,mask,saving = item
            added = mask & ~covered
            return (-int(added.sum()),-int(saving[added].sum()),route.km,route.id)
        route,mask,_ = min(legal,key=rank)
        covered |= mask
        chosen.append(route)
        used_h.add(route.hub)
        used_s.add(route.spoke)
    candidates = tuple(dict(id=i, location=point, count=n, role=role, active=i in active)
                       for role, group, active in (("Hub",hubs,used_h),("Spoke",spokes,used_s))
                       for i,point,n in group)
    base=replace(base,routes=tuple(chosen),candidates=candidates)
    stats = {"planning":len(orders),"potential":len(potential),
             "covered":int(covered.sum()),"routes":len(chosen),
             "pickup_clusters":len(set(origins)-{-1}),"dropoff_clusters":len(set(destinations)-{-1}),
             "pickup_noise":int((origins==-1).sum()),"dropoff_noise":int((destinations==-1).sum()),
             "cluster_flows":[{"pickup":int(a),"dropoff":int(b),"orders":int(n)}
                              for (a,b),n in Counter(zip(origins,destinations)).most_common()]}
    connections, savings = [], []
    route_by_id={r.id:r for r in chosen}
    for order in potential:
        match=match_order(base,order)
        if match.route_id:
            route=route_by_id[match.route_id]
            a=base.road(order.pickup,route.pickup)
            b=base.road(route.dropoff,order.dropoff)
            connections.append((a,b))
            savings.append(minutes(base.road(order.pickup,order.dropoff),18)+2-
                           (minutes(a,18)+minutes(b,18)+minutes(route.km,40)+6))
    if connections:
        stats["first_p90_km"],stats["last_p90_km"]=map(float,np.percentile(connections,90,axis=0))
        stats["nominal_air_faster"]=sum(s>0 for s in savings)
    return replace(base,diagnostics=stats)


def _site(point):
    return f"{point[0]:.8f},{point[1]:.8f}"


def covered_rows(network, orders):
    return [asdict(match_order(network,o)) for o in orders]
