"""Manual dispatch, workspace backups, and demand returned to planning."""
import io
import json
import re
import sqlite3
import zipfile
from hashlib import sha256
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from math import ceil, floor, isfinite

from .batch import resolve_restriction
from .cli import profile_extent
from .data import generate, planning_sample
from .examples import battery_pair
from .model import BUSINESS, LAST_ARRIVAL, MODES, START, Network, Order
from .network import build
from .sim import Engine
from .storage import MAX_SNAPSHOT, encode

DEMO_SOURCE = "demo"
SAMPLE_SOURCE = "battery-demo"
SOURCE_ID = re.compile(r"(?:demo|battery-demo|[a-f0-9]{64})")
MANUAL_ID = re.compile(r"M-[a-f0-9]{32}")
HASH_ID = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class ManualRecord:
    order: Order
    source: str
    network_version: str
    ordered_at: str
    normal: dict
    result: dict
    disturbance: dict | None
    chosen_mode: str
    delivered_at: str | None = None
    chosen_route_id: str | None = None

    def __post_init__(self):
        if not MANUAL_ID.fullmatch(self.order.id) or not SOURCE_ID.fullmatch(self.source):
            raise ValueError("手工订单编号或数据来源无效")
        if not HASH_ID.fullmatch(self.network_version):
            raise ValueError("航网版本无效")
        order = self.order
        ordered_at = datetime.fromisoformat(self.ordered_at)
        if (ordered_at.tzinfo is not None or order.arrival != ordered_at.hour * 60 + ordered_at.minute
                or not START <= order.arrival < LAST_ARRIVAL):
            raise ValueError("下单时间无效")
        if (order.source != "manual" or order.ready < order.arrival
                or (order.deadline is not None and order.deadline < order.arrival)
                or order.business not in BUSINESS or not isfinite(order.kg) or order.kg <= 0):
            raise ValueError("手工订单参数无效")
        for point in (order.pickup, order.dropoff):
            if len(point) != 2 or not -85 < point[0] < 85 or not -180 <= point[1] <= 180:
                raise ValueError("订单坐标无效")
        for result in (self.normal, self.result):
            if result["mode"] not in MODES:
                raise ValueError("配送建议无效")
            for key in ("ground_eta", "air_eta"):
                eta = result[key]
                if eta is not None and (not isinstance(eta, int) or not order.arrival <= eta):
                    raise ValueError("预计送达时间无效")
        if self.chosen_mode not in ("ground", "air") or self.result[f"{self.chosen_mode}_eta"] is None:
            raise ValueError("请选择可执行的配送方式")
        if self.chosen_route_id is not None and (self.chosen_mode != "air"
                or self.chosen_route_id != self.result.get("air_route_id")
                or self.chosen_route_id not in order.candidate_route_ids):
            raise ValueError("确认航段须与可执行的空地方案一致")
        if self.delivered_at is not None:
            arrival = datetime.fromisoformat(self.delivered_at)
            if arrival.tzinfo is not None or arrival < ordered_at:
                raise ValueError("送达时间不能早于下单时间")

    @classmethod
    def from_dict(cls, row):
        return cls(**{**row, "order": Order.from_dict(row["order"])})


class ManualStore:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS manual_orders (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, ordered_at TEXT NOT NULL, body TEXT NOT NULL
            )""")

    def list(self):
        with closing(sqlite3.connect(self.path)) as conn:
            data = conn.execute("SELECT body FROM manual_orders ORDER BY ordered_at DESC, id").fetchall()
        return [ManualRecord.from_dict(json.loads(row[0])) for row in data]

    def confirm(self, record):
        self.merge((record,))

    def complete(self, order_id, delivered_at):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT body FROM manual_orders WHERE id=?", (order_id,)).fetchone()
            if not row:
                raise ValueError("手工订单不存在")
            record = ManualRecord.from_dict(json.loads(row[0]))
            updated = replace(record, delivered_at=delivered_at)
            if record.delivered_at is not None and record != updated:
                raise ValueError("这笔订单已经完成")
            conn.execute("UPDATE manual_orders SET body=? WHERE id=?", (encode(asdict(updated)), order_id))
        return updated

    def merge(self, records):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            for record in records:
                order_id = record.order.id
                row = conn.execute("SELECT body FROM manual_orders WHERE id=?", (order_id,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO manual_orders VALUES (?,?,?,?)",
                                 (order_id, record.source, record.ordered_at, encode(asdict(record))))
                    continue
                current = ManualRecord.from_dict(json.loads(row[0]))
                if replace(current, delivered_at=None) != replace(record, delivered_at=None):
                    raise ValueError(f"手工订单 {order_id} 的记录与现有内容冲突")
                if not current.delivered_at and record.delivered_at:
                    conn.execute("UPDATE manual_orders SET body=? WHERE id=?",
                                 (encode(asdict(record)), order_id))
                elif current.delivered_at and record.delivered_at and current.delivered_at != record.delivered_at:
                    raise ValueError(f"手工订单 {order_id} 的送达时间与现有内容冲突")

    def export(self, network, source, label):
        snapshot = {"version": 2, "source": source, "source_label": label,
                    "network": asdict(network), "records": [asdict(row) for row in self.list()]}
        payload = encode(snapshot).encode("utf-8")
        if len(payload) > MAX_SNAPSHOT:
            raise ValueError("工作区备份超过 50 MB 解压上限")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("workspace.json", payload)
        return output.getvalue()

    def restore(self, payload):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entry = archive.getinfo("workspace.json")
            if entry.file_size > MAX_SNAPSHOT:
                raise ValueError("工作区备份超过 50 MB 解压上限")
            snapshot = json.loads(archive.read(entry))
        if snapshot["version"] != 2:
            raise ValueError("工作区备份版本不支持")
        source, label = snapshot["source"], snapshot["source_label"]
        if not SOURCE_ID.fullmatch(source) or not isinstance(label, str) or len(label) > 255:
            raise ValueError("数据来源无效")
        network = Network.from_dict(snapshot["network"])
        records = [ManualRecord.from_dict(row) for row in snapshot["records"]]
        self.merge(records)
        return network, source, label


def planning_orders(profile, records, source, count, include_manual=True, *, no_replacement=False):
    if count is not None and count < 1:
        raise ValueError("规划订单数须为正整数")
    low, high = profile_extent(profile)
    base = Network(profile.origin, (0, 0, 0, 0), ())
    (left, bottom), (right, top) = base.xy(low), base.xy(high)
    bounds = (floor(left * 2) / 2 - .5, ceil(right * 2) / 2 + .5,
              floor(bottom * 2) / 2 - .5, ceil(top * 2) / 2 + .5)
    area = replace(base, bounds=bounds)
    eligible = (row for row in records if row.delivered_at is not None and row.source == source
                and area.contains(row.order.pickup) and area.contains(row.order.dropoff))
    history = (sorted(eligible, key=lambda row: (row.delivered_at, row.order.id), reverse=True)[:count]
               if include_manual else [])
    if count is None:
        orders = planning_sample(profile)
    elif no_replacement:
        orders = planning_sample(profile, count - len(history))
    else:
        orders = generate(profile, count - len(history), 101, "planning")
    orders.extend(replace(row.order, route_id=None, eligibility="未匹配") for row in history)
    return orders, len(history)


def construct(profile, records, source, count, include_manual=True, *, no_replacement=False, **settings):
    orders, returned = planning_orders(profile, records, source, count, include_manual,
                                       no_replacement=no_replacement)
    network = build(orders, profile.origin, extent=profile_extent(profile), **settings)
    checksum = sha256()
    for order in orders:
        checksum.update((encode(order.input_dict()) + "\n").encode("utf-8"))
    diagnostics = {**network.diagnostics, "returned_orders": returned,
                   "planning_input_sha256": checksum.hexdigest()}
    return replace(network, diagnostics=diagnostics)


def manual_preview(network, order, disturbance=None, *, sample=False, ground_model=None, settings=None):
    run = battery_pair(False) if sample else Engine(network, ground_model=ground_model, settings=settings)
    if sample:
        # Keep the three background orders, and evaluate the fourth through the user-facing form.
        run.records["N4"]["status"] = "removed"
    checked = run.prepare_order(order)
    if disturbance and disturbance.kind != "demand":
        run.limit(*resolve_restriction(run, disturbance))
    run.advance(order.arrival)
    waiting = [row["order"] for row in run.records.values() if row["status"] in ("waiting", "unready")]
    result, _ = run.evaluate(checked, waiting, now=order.arrival)
    return result
