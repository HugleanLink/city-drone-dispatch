import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .data import KEYS, Profile, day_groups, planning_sample, prepare, profile_for_groups
from .deployment import deploy
from .model import Order
from .speed import fit_hourly


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_profile(path):
    path=Path(path)
    meta=read(path.with_name("profile.meta.json"))
    rows=pd.read_csv(path,dtype={"group":str,"biz_line":str})
    if not {"group","count",*KEYS}<=set(rows):
        raise ValueError("需求分布字段不完整")
    return Profile(rows,tuple(meta["origin"]),meta["report"])


def main():
    parser=argparse.ArgumentParser(description="城市物流无人机航网规划与空地协同调度系统")
    sub=parser.add_subparsers(dest="action",required=True)
    prep=sub.add_parser("prepare-data")
    prep.add_argument("--input",required=True,type=Path); prep.add_argument("--mapping",type=Path)
    prep.add_argument("--out",required=True,type=Path)
    net=sub.add_parser("build-network")
    net.add_argument("--profile",required=True,type=Path)
    net.add_argument("--orders",type=Path,help="可选：读取事先导出的历史规划订单")
    net.add_argument("--count",type=int,help="规划抽样数量，省略时使用完整历史来源")
    net.add_argument("--out",required=True,type=Path)
    serve=sub.add_parser("serve"); serve.add_argument("--port",default=8501,type=int); serve.add_argument("--workspace",type=Path)
    args=parser.parse_args()
    if args.action=="serve":
        from streamlit.web import cli as streamlit_cli
        os.environ["DRONE_DISPATCH_MODE"]="local"
        if args.workspace:
            os.environ["DRONE_DISPATCH_RUNS"]=str(args.workspace.resolve()/"runs")
        sys.argv=["streamlit","run",str(Path(__file__).parents[1]/"streamlit_app.py"),"--server.address","127.0.0.1","--server.port",str(args.port)]
        return streamlit_cli.main()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.action=="prepare-data":
        profile=prepare(args.input,read(args.mapping) if args.mapping else None)
        profile.rows.to_csv(args.out/"profile.csv.gz",index=False)
        save(args.out/"profile.meta.json",{"schema_version":1,"origin":profile.origin,"report":profile.report})
        print(profile.report)
    if args.action=="build-network":
        profile=load_profile(args.profile)
        profile=profile_for_groups(profile, day_groups(profile))
        if not {"pre_pickup", "reference_minutes"} <= set(profile.rows):
            parser.error("请先运行 prepare-data 以生成地面耗时字段")
        orders = ([Order.from_dict(o) for o in read(args.orders)] if args.orders else
                  planning_sample(profile, args.count))
        network=deploy(orders,profile.origin,profile_extent(profile))
        save(args.out/"network.json",asdict(network))
        save(args.out/"ground-speeds.json",asdict(fit_hourly(profile)))
        save(args.out/"road_graph.json",road_graph(network))
        pd.DataFrame(network.candidates).to_csv(args.out/"candidates.csv",index=False,encoding="utf-8-sig")
        print(network.diagnostics)
    return 0


def profile_extent(profile):
    s=profile.rows
    return [(min(s.pickup_latitude.min(),s.dropoff_latitude.min()),min(s.pickup_longitude.min(),s.dropoff_longitude.min())),
            (max(s.pickup_latitude.max(),s.dropoff_latitude.max()),max(s.pickup_longitude.max(),s.dropoff_longitude.max()))]


def road_graph(network):
    left,right,bottom,top=network.bounds
    x_range=range(round(left*2),round(right*2)+1)
    y_range=range(round(bottom*2),round(top*2)+1)
    def code(x,y):
        return f"{x}:{y}"
    nodes=[]
    edges=[]
    for x in x_range:
        for y in y_range:
            lat,lon=network.latlon(x/2,y/2)
            id=code(x,y)
            nodes.append(dict(id=id,x=x/2,y=y/2,lat=lat,lon=lon))
            for xx,yy in ((x+1,y),(x,y+1)):
                if xx in x_range and yy in y_range:
                    edges.append({"from":id,"to":code(xx,yy),"length_km":.5})
    return {"schema_version":1,"origin":network.origin,"bounds":network.bounds,
            "coordinate_system":"WGS84","step_km":.5,"nodes":nodes,"edges":edges}


if __name__=="__main__":
    raise SystemExit(main())
