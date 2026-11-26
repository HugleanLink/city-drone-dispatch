from dataclasses import asdict

import pytest

from src.deployment import runtime_routes
from src.model import FlightRules, Network, Order, Route, haversine
from src.operations import Dispatcher, Schedule, absolute_minute, joint, saved_schedule
from src.speed import HourlySpeeds
from src.workspace import DEMO_SOURCE, ManualRecord, ManualStore


POINT = (22.50, 113.90)
END = (22.50, 113.92)
STAMP = "2026-11-26T12:00"
VERSION = "a" * 64


def dispatcher(riders=(1, 1, 1), aircraft=1):
    route = Route("H1-S1", "H1", "S1", POINT, END, haversine(POINT, END), 0)
    network = Network(POINT, (-10, 10, -10, 10), (route,),
                      rules=FlightRules(order_min=1, access_km=1, report=True))
    return Dispatcher(network, HourlySpeeds(((4., 6.),) * 24, ()), riders, {"H1": aircraft})


def job(dispatch, i=1, stamp=STAMP):
    minute = absolute_minute(stamp) % 1440
    order = Order(f"M-{i:032x}", minute, minute + 8, None, POINT, END, "pinhaofan", 1, source="manual")
    return dispatch.job(order, stamp)


def test_aircraft_remains_occupied_after_parcel_is_available_and_next_order_reads_state():
    dispatch = dispatcher()
    first = job(dispatch)
    route = dispatch.routes["H1-S1"]
    first_plan = dispatch.air(first, route, Schedule(), first.arrival)
    aircraft = next(slot for slot in first_plan.slots if slot.resource.startswith("aircraft:"))
    assert aircraft.end > first_plan.air_times[2]
    state = Schedule().reserve(asdict(first_plan))
    second_plan = dispatch.air(job(dispatch, 2), route, state, first.arrival)
    next_flight = next(slot for slot in second_plan.slots if slot.resource.startswith("aircraft:"))
    assert next_flight.start >= aircraft.end


def test_joint_competes_for_real_resources_and_reports_infeasibility():
    dispatch = dispatcher()
    entries = [job(dispatch, i) for i in range(3)]
    state = Schedule()
    _, plans, report = joint(dispatch, entries, state, entries[0].arrival, population=4, generations=5)
    assert all(plans) and not report["fallback"]
    assert report["objectives"][0] == sum(plan.eta - entry.arrival for entry, plan in zip(entries, plans))
    slots = {}
    for plan in plans:
        for slot in plan.slots:
            slots.setdefault(slot.resource, []).append((slot.start, slot.end))
    for ranges in slots.values():
        ranges.sort()
        assert all(end <= start for (_, end), (start, _) in zip(ranges, ranges[1:]))
    assert state == Schedule()
    _, plans, report = joint(dispatcher((0, 0, 0)), entries, state, entries[0].arrival)
    assert report["fallback"] and plans == (None, None, None)


def test_confirmed_plan_is_atomic_and_crosses_midnight(tmp_path):
    dispatch = dispatcher(aircraft=0)
    first = job(dispatch, stamp="2026-11-26T23:59")
    ground, _, _ = dispatch.immediate(first, Schedule(), first.arrival)
    assert ground.eta // 1440 == first.arrival // 1440 + 1
    result = {"mode": "ground", "ground_eta": ground.eta - first.arrival + first.order.arrival, "air_eta": None}
    record = ManualRecord(first.order, DEMO_SOURCE, VERSION, first.ordered_at, result, result, None, "ground",
                          reservation=asdict(ground))
    store = ManualStore(tmp_path / "orders.sqlite3")
    saved, version = store.confirmed_state(DEMO_SOURCE, VERSION)
    assert not saved
    store.confirm_plans([record], DEMO_SOURCE, VERSION, version)
    with pytest.raises(ValueError, match="状态变更"):
        store.confirm_plans([record], DEMO_SOURCE, VERSION, version)
    confirmed, _ = store.confirmed_state(DEMO_SOURCE, VERSION)
    assert confirmed == [record]
    second = job(dispatch, 2, first.ordered_at)
    assert dispatch.ground(second, saved_schedule(confirmed), second.arrival).eta > ground.eta


def test_runtime_paths_use_only_open_hub_corridors_and_include_the_intermediate_hub():
    hub2 = (22.50, 113.91)
    pairs = [(0, 1)]
    routes = list(runtime_routes([POINT, hub2], [END], [1], [True, True], [True], pairs))
    cross = next(route for route in routes if route.hub == "H1")
    assert cross.via == (hub2,)
    assert cross.km == haversine(POINT, hub2) + haversine(hub2, END)
    local_only = list(runtime_routes([POINT, hub2], [END], [1], [True, True], [True], []))
    assert [route.hub for route in local_only] == ["H2"]
