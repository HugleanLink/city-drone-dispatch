from dataclasses import replace

from src.batch import Disturbance, run_batch
from src.calibration import fit
from src.data import prepare
from src.efficiency import operating_settings
from src.examples import mechanism_network
from src.model import Order
from src.sim import Engine


def profile():
    header = b"pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,customer_pay_time,fetch_time,arrived_time\n"
    route = mechanism_network().routes[0]
    fields = ",".join(map(str, (*route.pickup, *route.dropoff)))
    return prepare(header + (f"{fields},23:59,00:05,00:25\n" * 45).encode())


def test_checkout_after_midnight_without_truncating_resource_calendar():
    data = profile()
    network = mechanism_network()
    model = fit(data)
    route = network.routes[0]
    order = Order("late", 1439, 1439, 1474, route.pickup, route.dropoff, "waimai", 1.)
    settings = {"rider_hours": [1] + [0] * 23}
    run = Engine(network, [order], "ground", settings, ground_model=model)
    run.finish()
    assert run.records["late"]["first"] >= 1440
    assert run.records["late"]["actual"] > 1440
    assert all(slot.start >= 1440 for slots in run.cal.slots.values() for slot in slots)


def test_four_strategies_pair_totals_and_keep_plan_version():
    data = profile()
    model = fit(data)
    network = replace(mechanism_network(), routes=())
    settings = operating_settings(data, network, 2)
    settings["rider_hours"] = [2] * 24
    rows = run_batch(data, network, (201,), ("normal",), orders_per_day=2,
                     settings=settings, ground_model=model)
    assert len({row["input_sha256"] for row in rows}) == 1
    assert all(row["calibration"] == model.version and row["paired_orders"] == 2 for row in rows)
    assert all(row["saved_minutes"] == row["saved_rider_minutes"] == row["undelivered"] == 0 for row in rows)
    assert all(row["rider_minutes"] == sum(e["minutes"] for e in row["rider_breakdown"]) for row in rows)
    assert Disturbance("battery", "H01", 50, 1430, 1470).end == 1470


def test_research_aircraft_scenario_with_only_one_available_plane():
    data = profile()
    network = mechanism_network()
    rows = run_batch(data, network, (201,), ("aircraft",), orders_per_day=1,
                     settings={"aircraft": 1, "rider_hours": [2] * 24}, ground_model=fit(data))
    assert len(rows) == 4
    assert all(row["resource_event"]["count"] == 1 and row["completed"] == 1 for row in rows)
