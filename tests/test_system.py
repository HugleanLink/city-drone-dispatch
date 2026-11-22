import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.data import generate, prepare
from src.demo_data import synthetic_csv
from src.examples import battery_pair, mechanism_network, three_orders
from src.metrics import metrics, paired
from src.model import Order, match_order, minutes
from src.network import build
from src.sim import Engine
from src.storage import RunStore, encode


@pytest.fixture(scope="module")
def source():
    profile=prepare(synthetic_csv(14_000))
    planning=generate(profile,4000,101,"planning")
    network=build(planning,profile.origin)
    return profile,planning,network


def test_generation_is_seeded_and_nested(source):
    profile,planning,_=source
    assert profile.report["输入"]==14_000
    assert len(planning)==4000
    assert planning==generate(profile,4000,101,"planning")
    normal=generate(profile,201,201)
    high=generate(profile,302,201,"heavy")
    assert high[:201]==normal
    assert len({o.id for o in high})==302
    assert all(600<=o.arrival<1200 and o.ready<=o.deadline for o in high)


def test_input_validations():
    data=b"pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,customer_pay_time,package_price\n0,0,0,.01,12:00,\n91,0,0,.01,12:00,10\n"
    p=prepare(data)
    assert p.report["输入"]==2 and p.report["坐标异常"]==1
    assert p.report["窗口内可用"]==1
    with pytest.raises(ValueError,match="缺少"):
        prepare(b"test\n0\n")


def test_network_reproducible_and_within_constraints(source):
    p,planning,network=source
    assert network==build(planning,p.origin)
    assert network.routes
    assert len(network.routes)<=6
    assert len({r.hub for r in network.routes})<=3
    assert len({r.spoke for r in network.routes})<=4
    assert all(.5<=r.km<=4 for r in network.routes)
    assert sum(bool(match_order(network,o).route_id) for o in planning)==network.diagnostics["covered"]
    assert minutes(2,40)==3


def test_preview_read_only_and_submit_idempotent():
    net=mechanism_network()
    route=net.routes[0]
    run=Engine(net)
    order=Order("manual",600,602,635,route.pickup,route.dropoff,"waimai",.7)
    sha=encode(run.snapshot())
    guide=run.preview(order)
    assert guide["air_eta"] and guide["ground_eta"]
    assert encode(run.snapshot())==sha
    assert run.add(order,"req")==run.add(order,"req")=="manual"
    run.advance(650)
    assert run.records["manual"]["status"]=="delivered"
    assert len(run.records)==1


def test_calendar_and_four_policies(source):
    _,planning,net=source
    selected=[replace(o,arrival=600+i//3,ready=600+i//3,deadline=630+i//3) for i,o in enumerate(planning[:15])]
    for strategy in ("ground","fixed","dynamic","lookahead"):
        run=Engine(net,selected,strategy)
        run.advance(1320)
        for resource,slots in run.cal.slots.items():
            ordered=sorted(slots,key=lambda x:(x.start,x.end))
            assert all(a.end<=b.start for a,b in zip(ordered,ordered[1:])),resource
        assert metrics(run)["arrivals"]==15
        assert all(r["status"] in ("incomplete","delivered") for r in run.records.values())


def test_lookahead_protects_two_punctual_orders():
    direct=three_orders("dynamic")
    planned=three_orders("lookahead")
    for run in (direct,planned):
        run.advance(760)
    assert metrics(direct)["on_time"]==1
    assert metrics(planned)["on_time"]==2
    assert [e["id"] for e in planned.log if e["type"]=="dispatch"][:2]==["B","C"]


def test_restriction_restores_and_does_not_cancel_reservations():
    normal=battery_pair(False)
    disturbed=battery_pair(True)
    disturbed.advance(720)
    impediment=[e for e in disturbed.log if e["type"]=="limit"][-1]
    assert len(impediment["resources"])==3
    for id in impediment["resources"]:
        assert disturbed.cal.limits[id]
    assert len([s for slots in disturbed.cal.slots.values() for s in slots if s.order=="opening"])==8
    normal.advance(770)
    disturbed.advance(770)
    assert any(normal.records[id]["mode"]!=disturbed.records[id]["mode"] or
               normal.records[id].get("eta")!=disturbed.records[id].get("eta") for id in normal.records)
    disturbed.advance(780)
    assert not any(disturbed.cal.limits.values())


def test_run_package_restore(tmp_path):
    first=RunStore(tmp_path/"first")
    second=RunStore(tmp_path/"second")
    run=three_orders("lookahead")
    run.advance(723)
    clone=second.import_bundle(first.export(run))
    assert clone.run_id!=run.run_id
    for version in (run,clone):
        version.advance(760)
    assert metrics(run)==metrics(clone)
    assert clone.log==run.log
    assert clone.records==run.records


def test_stats_omit_missing_pairs():
    rows=[{"strategy":s,"scenario":"normal","seed":1,"on_time_rate":r}
          for s,r in [("ground",.5),("lookahead",.75)]]
    assert paired(rows,"on_time_rate","ground")[0]["均差 (pp)"]==25
