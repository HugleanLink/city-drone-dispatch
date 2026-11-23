import json
from dataclasses import replace
from threading import Event

import pytest

from src.diagnostics import NON_ADDITIVE, diagnostic_tables, diagnostics
from src.metrics import metrics
from src.model import Network, Order, Route, match_order
from src.sim import Engine
from src.storage import RunStore, encode
from src.workspace import manual_preview


def dual_hub():
    base = Network((30.21, 120.18), (-4, 4, -4, 4), ())
    pickup, dropoff = base.latlon(-1.35, -1.35), base.latlon(1.35, 1.35)
    alternate = base.latlon(-1.25, -1.35)
    routes = (
        Route("H01-S01", "H01", "S01", pickup, dropoff, base.straight(pickup, dropoff), 0),
        Route("H02-S01", "H02", "S01", alternate, dropoff, base.straight(alternate, dropoff), 0),
    )
    net = replace(base, routes=routes, rider_starts=(pickup, dropoff, pickup, dropoff))
    return net, Order("dual", 720, 720, None, pickup, dropoff, "waimai", 1.0)


def assert_calendar(run):
    for resource, slots in run.cal.slots.items():
        assert all(a.end <= b.start for a, b in zip(slots, slots[1:])), resource
    dispatched = [event["id"] for event in run.log if event["type"] == "dispatch"]
    assert len(dispatched) == len(set(dispatched))
    for resource, (end, _) in run.cal.tails.items():
        if slots := run.cal.slots.get(resource):
            assert slots[-1].end == end


@pytest.mark.parametrize(("strategy", "expected_mode", "expected_route"),
                         [("ground", "ground", None), ("fixed", "air", "H01-S01"),
                          ("dynamic", "air", "H02-S01"), ("lookahead", "air", "H02-S01")])
def test_blocked_preferred_hub(strategy, expected_mode, expected_route):
    net, order = dual_hub()
    matched = match_order(net, order)
    assert matched.route_id == "H01-S01"
    assert matched.candidate_route_ids == ("H01-S01", "H02-S01")
    run = Engine(net, [order], strategy, {"riders": 1, "aircraft": 1}, start=720)
    run.limit("aircraft", "H01", 1, 720, 780)
    run.finish()
    actual = run.records[order.id]
    assert actual["mode"] == expected_mode
    assert actual["executed_route_id"] == expected_route and actual["order"].route_id == "H01-S01"
    measures, details = diagnostics(run)
    elapsed = actual["actual"] - order.arrival
    assert sum(v for k, v in details[order.id]["stages"].items() if k not in NON_ADDITIVE) == elapsed
    assert sum(h["worked_minutes"] for h in measures["rider_hourly"]) == metrics(run)["rider_minutes"]
    if expected_route == "H02-S01":
        assert measures["reselected_route_orders"] == 1
        assert measures["route_examples"][0]["minutes_saved"] > 0
    assert_calendar(run)


def test_preview_and_cross_midnight_resource_fallback(tmp_path):
    net, order = dual_hub()
    order = replace(order, arrival=1439, ready=1439)
    settings = {"riders": 1, "aircraft_by_hub": {"H01": 0, "H02": 1}}
    result = manual_preview(net, order, settings=settings)
    assert result["preferred_route_id"] == "H01-S01" and result["air_route_id"] == "H02-S01"
    run = Engine(net, [order], "dynamic", settings)
    run.advance(1444)
    store = RunStore(tmp_path)
    clone = store.import_bundle(store.export(run))
    for candidate in (run, clone):
        candidate.finish()
        actual = candidate.records[order.id]
        assert actual["actual"] > 1440 and actual["executed_route_id"] == "H02-S01"
        assert_calendar(candidate)
    assert run.records == clone.records and run.log == clone.log
    summary, orders = diagnostics(clone)
    table = diagnostic_tables([{"strategy": "dynamic", "seed": 1, "scenario": "normal",
                                "orders": orders, **summary}])["order_diagnostics"][0]
    assert table["candidate_route_ids"] == "H01-S01,H02-S01"
    assert table["executed_route_id"] == "H02-S01"
    assert summary["route_examples"] == []  # No valid ETA for the unavailable hub.


@pytest.mark.parametrize("legs", [0, 1, 2])
def test_ground_fallback_with_no_aircraft_and_reproducible_assignments(legs):
    net, order = dual_hub()
    net = replace(net, routes=net.routes[:legs])
    orders = [replace(order, id=str(i), arrival=720+i, ready=720+i) for i in range(4)]
    runs = [Engine(net, orders, "lookahead", {"riders": 1, "aircraft": 0}, start=720) for _ in range(2)]
    for run in runs:
        run.finish()
        assert all(r["mode"] == "ground" and r["executed_route_id"] is None for r in run.records.values())
        assert_calendar(run)
    assert runs[0].records == runs[1].records and runs[0].cal.slots == runs[1].cal.slots


def test_cancelled_run_can_resume_without_duplicate_bookings():
    net, order = dual_hub()
    run = Engine(net, [order], "dynamic", start=720)
    cancelled = Event()
    cancelled.set()
    run.finish(cancelled.is_set)
    assert len(run.log) == 0 and not run.cal.slots
    cancelled.clear()
    run.finish(cancelled.is_set)
    assert run.records[order.id]["actual"] is not None
    restored = Engine.restore(json.loads(encode(run.snapshot())))
    assert run.records == restored.records
    assert_calendar(restored)
