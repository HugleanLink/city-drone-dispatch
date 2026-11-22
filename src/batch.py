import csv
import hashlib
import json
from dataclasses import asdict

from .data import generate
from .metrics import metrics
from .model import END, STRATEGIES, match_order
from .sim import Engine

SCENARIOS = {"normal":"常态","heavy":"高需求","battery":"备用电池受限",
             "aircraft":"机队受限","pad":"起降位受限","demand":"局部订单激增",
             "pads2":"双起降位","riders3":"减少骑手"}


def target_hub(network,planning):
    links={r.id:r.hub for r in network.routes}
    counts={r.hub:0 for r in network.routes}
    for o in planning:
        order=match_order(network,o)
        if order.route_id:
            counts[links[order.route_id]]+=1
    return min(counts,key=lambda hub:(-counts[hub],hub),default=None)


def run_batch(profile,network,seeds,scenarios,*,hub=None,settings=None,cancelled=lambda:False,progress=lambda row:None):
    output=[]
    for seed in seeds:
        for scenario in scenarios:
            if cancelled():
                return output
            heavy=scenario!="normal"
            orders=generate(profile,302 if heavy else 201,seed,"heavy" if heavy else "operation")
            extra=[]
            if scenario=="demand":
                count=sum(720<=o.arrival<750 for o in orders)
                extra=generate(profile,(count+1)//2,seed+100_000,window=(720,750),prefix="D")
            run_settings={**(settings or {})}
            if scenario=="pads2":
                run_settings["pads"]=2
            if scenario=="riders3":
                run_settings["riders"]=3
            # Resolve once from initial resource IDs and replay identical restrictions in every policy.
            prototype=Engine(network,settings=run_settings)
            hub=hub or min(prototype.hubs,default=None)
            event=None
            if scenario in ("battery","aircraft","pad") and hub:
                route=next(r for r in network.routes if r.hub==hub)
                site=(route.hub_site or route.hub) if scenario=="pad" else hub
                count,start,end={"battery":(3,720,740),"aircraft":(2,720,740),"pad":(1,730,745)}[scenario]
                ids=prototype.resources(scenario,site)[:count]
                if len(ids)!=count:
                    raise ValueError(f"场景 {scenario} 的初始资源不足")
                event=(scenario,site,count,start,end,ids)
            digest=hashlib.sha256(json.dumps({"orders":[asdict(o) for o in orders+extra],"event":event,
                                             "settings":run_settings,"network":asdict(network)},
                                             sort_keys=True,separators=(",",":")).encode()).hexdigest()
            for strategy in STRATEGIES:
                if cancelled():
                    return output
                run=Engine(network,orders+extra,strategy,run_settings)
                if event:
                    run.limit(*event)
                run.advance(END,cancelled)
                if cancelled():
                    return output
                row={"seed":seed,"scenario":scenario,"strategy":strategy,"input_sha256":digest,**metrics(run)}
                output.append(row)
                progress(row)
    return output


def write_results(rows,path):
    if not rows:
        return
    fields=[k for k,v in rows[0].items() if not isinstance(v,dict)]
    with path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
