import numpy as np
from .model import START


def resource_usage(run, resource_ids):
    occupied=sum(max(0,min(s.end,run.now)-max(s.start,START))
                 for id in resource_ids for s in run.cal.slots.get(id,()))
    # Restrictions are reconstructed from the applied events, including restorations.
    exclusions={}
    stops={e["id"]:e["minute"] for e in run.log if e["type"]=="restore"}
    for e in run.log:
        if e["type"]=="limit":
            for id in set(e["resources"])&resource_ids:
                exclusions.setdefault(id,set()).update(range(e["minute"],stops.get(e["id"],run.now)))
    protected=0
    for id,blocked in exclusions.items():
        occupied_times={m for s in run.cal.slots.get(id,()) for m in range(max(s.start,START),min(s.end,run.now))}
        protected+=len(blocked-occupied_times)
    available=len(resource_ids)*max(0,run.now-START)-protected
    return {"used": occupied, "available": available, "ratio": ratio(occupied, available)}


def metrics(run):
    records=[r for r in run.records.values() if r["status"] not in ("scheduled","removed")]
    completed=[r for r in records if r["actual"] is not None]
    due = [r for r in records if r["order"].deadline is not None]
    due_completed = [r for r in due if r["actual"] is not None]
    timely = [r for r in due_completed if r["actual"] <= r["order"].deadline]
    ground_due = sum(r["order"].ground_only for r in due)
    ground=[r for r in records if r["order"].ground_only]
    ground_timely=sum(r["order"].ground_only for r in timely)
    elapsed=[r["actual"]-r["order"].arrival for r in completed]
    rider_minutes={}
    for resource,slots in run.cal.slots.items():
        if resource.startswith("rider:"):
            for s in slots:
                rider_minutes[s.activity]=rider_minutes.get(s.activity,0)+max(0,min(s.end,run.now)-max(s.start,START))
    windows={}
    for group in ("aircraft","pad","swap","charge"):
        resource_ids={id for r in run.hubs.values() for id in r.get(group,[])}
        if group=="pad":
            resource_ids={id for ids in run.pads.values() for id in ids}
        windows[group] = resource_usage(run, resource_ids)
    wait=[r["first"]-r["order"].ready for r in records if r.get("first") is not None]
    station_handoff={s.order:s.end for slots in run.cal.slots.values() for s in slots if s.activity=="首段接驳"}
    launch_wait=[s.start-station_handoff[s.order] for slots in run.cal.slots.values()
                 for s in slots if s.activity=="载货起飞" and s.start<=run.now and s.order in station_handoff]
    return {"arrivals":len(records),"completed":len(completed),"on_time":len(timely),
            "ground_only":len(ground),"ground_on_time":ground_timely,
            "deadline_orders": len(due), "ground_deadline_orders": ground_due,
            "on_time_rate":ratio(len(timely),len(due)),"ground_on_time_rate":ratio(ground_timely,ground_due),
            "delivered_on_time_rate":ratio(len(timely),len(due_completed)),
            "completion_rate":ratio(len(completed),len(records)),
            "mean_minutes":float(np.mean(elapsed)) if elapsed else None,
            "p50_minutes":float(np.median(elapsed)) if elapsed else None,
            "p90_minutes":float(np.percentile(elapsed,90)) if elapsed else None,
            "mean_wait":float(np.mean(wait)) if wait else None,
            "mean_launch_wait":float(np.mean(launch_wait)) if launch_wait else None,
            "air":sum(r["mode"]=="air" for r in records),"ground":sum(r["mode"]=="ground" for r in records),
            "switched":sum(r.get("ever_air",False) and r["mode"]=="ground" for r in records),
            "pending":sum(r["status"] in ("waiting","unready") for r in records),
            "assigned":sum(r["status"]=="assigned" for r in records),
            "rider_minutes":sum(rider_minutes.values()),"rider_detail":rider_minutes,"resources":windows}


def ratio(a,b):
    return a/b if b else None


def paired(rows, metric, baseline):
    by_seed={}
    for row in rows:
        by_seed.setdefault((row["scenario"],row["seed"]),{})[row["strategy"]]=row
    rng=np.random.default_rng(9001)
    results=[]
    for scene in sorted({k[0] for k in by_seed}):
        differences=[100*(v["lookahead"][metric]-v[baseline][metric])
                     for (scenario,_),v in by_seed.items()
                     if scenario==scene and {"lookahead",baseline}<=v.keys()
                     and v["lookahead"][metric] is not None and v[baseline][metric] is not None]
        if differences:
            mean=np.mean(rng.choice(differences,size=(2000,len(differences)),replace=True),axis=1)
            low,high=np.percentile(mean,[2.5,97.5])
            results.append({"场景":scene,"基线":baseline,"指标":metric,"有效日":len(differences),
                            "均差 (pp)":float(np.mean(differences)),"95% 下限":float(low),"95% 上限":float(high)})
    return results
