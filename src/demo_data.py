"""Generate the public dataset without input records or real order coordinates."""
from pathlib import Path
import gzip
import numpy as np
import pandas as pd
from .data import rng

PUBLIC_COUNT = 592_667


def synthetic_csv(count=PUBLIC_COUNT):
    random = rng(52284, 0)
    # A purpose-built simulated district with mixed local and cross-district flows.
    centers=np.array([[-4,-3],[-3,3],[0,0],[3,-3],[4,3],[1,5]])
    cluster=random.choice(len(centers),count,p=[.15,.18,.22,.15,.19,.11])
    dest=(cluster + random.choice([0,1,2,3,4,5],count,p=[.28,.19,.13,.15,.14,.11]))%len(centers)
    pickup=np.round((centers[cluster]+random.normal(0,.7,(count,2)))*2)/2
    drop=np.round((centers[dest]+random.normal(0,.85,(count,2)))*2)/2
    # Anchor only supplies a map reference; locations above do not come from a CSV.
    latitude,longitude=30.21,120.18
    scale=111*np.cos(np.deg2rad(latitude))
    meal=random.choice(3,count,p=[.40,.31,.29])
    clock=np.minimum(1199,np.maximum(600,np.round(random.normal(np.array([725,1110,925])[meal],np.array([40,45,110])[meal],count)).astype(int)))
    fetch=clock+random.integers(5,18,count)
    deliver=fetch+random.integers(8,42,count)
    def hm(series):
        return [f"{t//60%24:02}:{t%60:02}" for t in series]
    data=pd.DataFrame({
        "dt_seq_id":random.integers(1,8,count),
        "order_seq_id":np.arange(1,count+1),
        "pickup_latitude":np.round(latitude+pickup[:,1]/111,7),
        "pickup_longitude":np.round(longitude+pickup[:,0]/scale,7),
        "dropoff_latitude":np.round(latitude+drop[:,1]/111,7),
        "dropoff_longitude":np.round(longitude+drop[:,0]/scale,7),
        "customer_pay_time":hm(clock),
        "fetch_time":hm(fetch),
        "arrived_time":hm(deliver),
        "package_price":np.round(random.lognormal(3.5,.6,count),2),
        "biz_line":random.choice(["waimai","pinhaofan","shangou","health"],count,p=[.75,.1,.12,.03]),
    })
    return data.to_csv(index=False,lineterminator="\n").encode()


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("data/demo_orders.csv.gz"))
    parser.add_argument("--count",type=int,default=PUBLIC_COUNT)
    args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(gzip.compress(synthetic_csv(args.count),compresslevel=6,mtime=0))
    print(f"{args.count:,} records, {args.out.stat().st_size:,} compressed bytes")
