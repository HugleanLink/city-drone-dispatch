import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .data import KEYS, Profile, generate, prepare
from .model import Network, Order
from .network import build, covered_rows
from .batch import SCENARIOS, run_batch, write_results


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
    gen=sub.add_parser("generate")
    gen.add_argument("--profile",required=True,type=Path); gen.add_argument("--purpose",choices=["planning","operation","heavy"],required=True)
    gen.add_argument("--seed",required=True,type=int); gen.add_argument("--count",required=True,type=int)
    gen.add_argument("--out",required=True,type=Path)
    net=sub.add_parser("build-network")
    net.add_argument("--orders",required=True,type=Path); net.add_argument("--profile",type=Path)
    net.add_argument("--out",required=True,type=Path)
    bat=sub.add_parser("batch")
    bat.add_argument("--profile",required=True,type=Path); bat.add_argument("--network",required=True,type=Path)
    bat.add_argument("--seeds",default="201:230"); bat.add_argument("--scenarios",nargs="+",choices=SCENARIOS,default=list(SCENARIOS))
    bat.add_argument("--out",required=True,type=Path)
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
    if args.action=="generate":
        orders=generate(load_profile(args.profile),args.count,args.seed,args.purpose)
        save(args.out/"orders.json",[asdict(o) for o in orders])
        pd.DataFrame([asdict(o) for o in orders]).to_csv(args.out/"orders.csv",index=False,encoding="utf-8-sig")
    if args.action=="build-network":
        orders=[Order.from_dict(o) for o in read(args.orders)]
        profile=load_profile(args.profile) if args.profile else None
        network=build(orders,profile.origin if profile else None,extent=profile_extent(profile) if profile else None)
        save(args.out/"network.json",asdict(network))
        save(args.out/"coverage.json",covered_rows(network,orders))
        print(network.diagnostics)
    if args.action=="batch":
        profile=load_profile(args.profile)
        network=Network.from_dict(read(args.network))
        seed=args.seeds.split(":")
        seeds=range(int(seed[0]),int(seed[-1])+1)
        rows=run_batch(profile,network,seeds,args.scenarios,progress=lambda row:print(row["scenario"],row["seed"],row["strategy"],flush=True))
        save(args.out/"batch.json",rows)
        write_results(rows,args.out/"batch.csv")
    return 0


def profile_extent(profile):
    s=profile.rows
    return [(min(s.pickup_latitude.min(),s.dropoff_latitude.min()),min(s.pickup_longitude.min(),s.dropoff_longitude.min())),
            (max(s.pickup_latitude.max(),s.dropoff_latitude.max()),max(s.pickup_longitude.max(),s.dropoff_longitude.max()))]


if __name__=="__main__":
    raise SystemExit(main())
