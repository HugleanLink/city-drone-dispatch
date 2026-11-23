import json
from io import BytesIO

import streamlit as st
from dataclasses import asdict, replace
from datetime import time
from pathlib import Path
from threading import Event

import pytest
from streamlit.testing.v1 import AppTest

from src.batch import Disturbance, digest, load_results, run_batch, summarize_results, write_results
from src.data import planning_sample, prepare, source_counts, split_sources
from src.demo_data import synthetic_csv
from src.examples import mechanism_network
from src.model import STRATEGIES, Network, Order
from src.workspace import DEMO_SOURCE, ManualRecord, ManualStore, construct, manual_preview


@pytest.fixture(scope="module")
def profile():
    return prepare(synthetic_csv(12_000))


def manual_order():
    route = mechanism_network().routes[0]
    return Order("M-" + "1" * 32, 720, 720, 746, route.pickup, route.dropoff, "waimai", 1., source="manual")


def confirmed_record(network, order):
    result = manual_preview(network, order)
    mode = "ground" if result["ground_eta"] is not None else "air"
    return ManualRecord(order, DEMO_SOURCE, digest(asdict(network)), "2026-01-01T12:00", result, result, None, mode)


def test_manual_preview_uses_same_order_for_battery_comparison():
    network, order = mechanism_network(), manual_order()
    before = manual_preview(network, order, sample=True)
    d = Disturbance("battery", "H01", 50, 720, 780)
    after = manual_preview(network, order, d, sample=True)
    assert before["order"] == after["order"] == order.id
    assert (before["mode"], before["air_eta"]) == ("air", 736)
    assert (after["mode"], after["ground_eta"]) == ("ground", 743)
    assert manual_preview(network, order, sample=True) == before


def test_backup_restore_updates_completed_once_and_returns_to_planning(profile, tmp_path):
    network = construct(profile, [], DEMO_SOURCE, 100)
    order = replace(manual_order(), pickup=profile.origin, dropoff=network.latlon(1, 0))
    store = ManualStore(tmp_path / "first.sqlite3")
    record = confirmed_record(network, order)
    store.confirm(record)
    store.confirm(record)
    with pytest.raises(ValueError, match="送达时间"):
        store.complete(order.id, "2026-01-01T11:59")
    assert store.list()[0].delivered_at is None

    second = ManualStore(tmp_path / "second.sqlite3")
    second.restore(store.export(network, DEMO_SOURCE, "公开模拟"))
    finished = store.complete(order.id, "2026-01-02T00:10")
    store.confirm(record)
    bundle = store.export(network, DEMO_SOURCE, "公开模拟")
    assert second.restore(bundle)[0] == network
    second.restore(bundle)
    assert second.list() == [finished]

    rebuilt = construct(profile, second.list(), DEMO_SOURCE, 100)
    assert rebuilt.diagnostics["planning"] == 100
    assert rebuilt.diagnostics["returned_orders"] == 1
    assert rebuilt.diagnostics["planning_input_sha256"] != network.diagnostics["planning_input_sha256"]
    assert construct(profile, second.list(), "a" * 64, 100).diagnostics["returned_orders"] == 0


def test_batch_matching_hashes_import_and_cancel(profile, tmp_path):
    net = construct(profile, [], DEMO_SOURCE, 100)
    disturbance = Disturbance("demand", "", 50, 600, 1200)
    rows = run_batch(profile, net, (201, 202), ("normal",), orders_per_day=4, disturbance=disturbance)
    assert len(rows) == 8 and all(row["arrivals"] == 6 for row in rows)
    assert sum(source_counts(profile).values()) == 12_000
    for seed in (201, 202):
        group = [row for row in rows if row["seed"] == seed]
        assert {row["strategy"] for row in group} == set(STRATEGIES)
        assert len({row["input_sha256"] for row in group}) == 1
    summary = summarize_results(load_results(json.dumps(rows).encode()))
    assert summary[0]["arrivals"] == 12
    assert summary[0]["on_time_rate_delta"] == 0
    for item in summary:
        same = [row for row in rows if row["strategy"] == item["strategy"]]
        assert item["on_time_rate"] == sum(row["on_time"] for row in same) / 12
    path = tmp_path / "batch.csv"
    write_results(rows, path)
    assert summarize_results(load_results(path.read_bytes(), csv_file=True)) == summary

    stop = Event()
    partial = run_batch(profile, net, (201,), ("normal",), orders_per_day=4,
                        cancelled=stop.is_set, progress=lambda row: stop.set())
    assert [row["strategy"] for row in partial] == ["ground"]


def test_foreign_profile_build_switches_the_manual_coordinates():
    payload = (b"pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,customer_pay_time\n"
               b"35.68,139.750,35.69,139.77,12:00\n")
    net = construct(prepare(payload), [], "b" * 64, 25)
    assert net.origin == (35.685, 139.76)
    assert net.diagnostics["planning"] == 25
    order = replace(manual_order(), pickup=(35.68, 139.750), dropoff=(35.69, 139.77))
    assert manual_preview(net, order)["ground_eta"] is not None
    with pytest.raises(ValueError, match="范围"):
        manual_preview(net, manual_order())


def test_manual_source_split_uses_each_input_record_at_most_once():
    payload = (b"pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,customer_pay_time,weight_kg\n"
               + b"35.68,139.75,35.69,139.77,12:00,0.7\n" * 10
               + b"35.68,139.75,35.69,139.77,12:30,2.2\n" * 8)
    plan, operation = split_sources(prepare(payload), 12, 6)
    planned, operated = planning_sample(plan), planning_sample(operation)
    assert (len(planned), len(operated)) == (12, 6)
    assert sum(order.kg == .7 for order in planned + operated) == 10
    assert len(planning_sample(plan, 3)) == 3


def test_single_page_manual_delivery(monkeypatch):
    entrypoint = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(entrypoint, default_timeout=20).run()
    assert not app.exception
    assert not app.button
    assert app.session_state.network is None

    upload = BytesIO(b"pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,customer_pay_time,weight_kg\n"
                     + b"35.68,139.75,35.69,139.77,12:00,1.1\n" * 6)
    upload.file_id, upload.name = "test-file", "test.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app.run()
    assert not app.exception
    assert not {"字段映射与编码", "工作区备份与恢复", "导入多日对照结果", "查看共同输入哈希"} & {
        expander.label for expander in app.expander
    }
    next(box for box in app.checkbox if box.label == "使用全部规划来源").check().run()
    next(widget for widget in app.number_input if widget.label == "包裹最大重量 kg").set_value(1.05).run()
    next(widget for widget in app.button if widget.label == "分析并构建航网").click().run()
    assert not app.exception
    net = app.session_state.network
    assert net.diagnostics["planning"] == 6
    assert net.diagnostics["potential"] == 0
    assert net.rules.weight == 1.05 and Network.from_dict(asdict(net)) == net
    assert any(heading.value == "运营策略对照" for heading in app.subheader)

    def button(label):
        return next(widget for widget in app.button if widget.label == label)

    button("计算配送建议").click().run()
    assert not app.exception
    assert app.session_state.draft["result"]["mode"] == "ground"
    button("确认送单方式").click().run()
    assert app.session_state.manual_store.list()[0].source == app.session_state.profile_key
    assert button("登记实际送达").disabled
    next(widget for widget in app.time_input if widget.label == "实际送达时间").set_value(time(12, 28)).run()
    button("登记实际送达").click().run()
    assert not app.exception
    assert len(app.session_state.manual_store.list()) == 1
    assert app.session_state.manual_store.list()[0].delivered_at.endswith("T12:28")
