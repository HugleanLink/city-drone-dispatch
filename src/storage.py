import csv
import io
import json
import re
import sqlite3
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .sim import Engine

RUN_ID = re.compile(r"[0-9a-f]{12}")
MAX_SNAPSHOT = 50*1024*1024


def encode(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


class RunStore:
    def __init__(self,root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)

    def path(self,id):
        if not RUN_ID.fullmatch(id):
            raise ValueError("运行编号无效")
        return self.root/id/"orders.sqlite3"

    def save(self,run):
        path=self.path(run.run_id)
        path.parent.mkdir(parents=True,exist_ok=True)
        snapshot=run.snapshot()
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, version INTEGER, updated TEXT, snapshot TEXT);
                CREATE TABLE IF NOT EXISTS orders (id TEXT PRIMARY KEY, state TEXT, body TEXT);
                CREATE TABLE IF NOT EXISTS decisions (seq INTEGER PRIMARY KEY, order_id TEXT, minute INTEGER, body TEXT);
                CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, minute INTEGER, body TEXT);
                CREATE TABLE IF NOT EXISTS reservations (resource TEXT, start INTEGER, end INTEGER, body TEXT);
                CREATE INDEX IF NOT EXISTS decisions_by_order ON decisions(order_id, minute);
            """)
            conn.execute("BEGIN IMMEDIATE")
            current=conn.execute("SELECT version FROM runs WHERE id=?",(run.run_id,)).fetchone()
            if current and current[0]!=getattr(run,"saved_version",None):
                raise ValueError("运行状态已由另一个窗口更新，请重新载入")
            conn.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?)",
                         (run.run_id,run.version,datetime.now(timezone.utc).isoformat(),encode(snapshot)))
            conn.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?)",
                ((id,record["status"],encode(record)) for id,record in snapshot["records"].items()
                 if record["status"] not in ("scheduled","removed")))
            for table, values in (
                ("decisions",((i,d["order"],d["minute"],encode(d)) for i,d in enumerate(run.decisions))),
                ("events",((i,e["minute"],encode(e)) for i,e in enumerate(run.log))),
            ):
                placeholders="?,?,?,?" if table=="decisions" else "?,?,?"
                conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})",values)
            conn.execute("DELETE FROM reservations")
            conn.executemany("INSERT INTO reservations VALUES (?,?,?,?)",
                ((resource,slot["start"],slot["end"],encode(slot)) for resource,slots in snapshot["slots"].items() for slot in slots))

        run.saved_version=run.version

    def load(self,id):
        path=self.path(id)
        if not path.is_file():
            raise ValueError("运行记录不存在")
        with closing(sqlite3.connect(path)) as conn:
            row=conn.execute("SELECT snapshot FROM runs WHERE id=?",(id,)).fetchone()
        if not row:
            raise ValueError("运行记录不完整")
        run=Engine.restore(json.loads(row[0]))
        run.saved_version=run.version
        return run

    def list(self):
        return sorted((d.name for d in self.root.iterdir()
                       if RUN_ID.fullmatch(d.name) and (d/"orders.sqlite3").is_file()),reverse=True)

    def export(self,run):
        self.save(run)
        output=io.BytesIO()
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("snapshot.json",encode(run.snapshot()))
            z.writestr("events.jsonl","\n".join(encode(e) for e in run.log))
            z.writestr("decisions.jsonl","\n".join(encode(e) for e in run.decisions))
            from .metrics import metrics
            z.writestr("metrics.json",encode(metrics(run)))
            rows=io.StringIO()
            writer=csv.writer(rows)
            writer.writerow(["order_id","mode","resource","activity","planned_start","planned_end","actual_elapsed"])
            for resource,slots in run.cal.slots.items():
                for slot in slots:
                    if slot.order in run.records:
                        mode=run.records[slot.order]["mode"]
                        actual=max(0,min(slot.end,run.now)-slot.start)
                        writer.writerow([slot.order,mode,resource,slot.activity,slot.start,slot.end,actual])
            z.writestr("calibration_stages.csv",rows.getvalue().encode("utf-8-sig"))
            z.write(self.path(run.run_id),"orders.sqlite3")
        return output.getvalue()

    def import_bundle(self,payload):
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            item=z.getinfo("snapshot.json")
            if item.file_size>MAX_SNAPSHOT:
                raise ValueError("运行包超过 50 MB 解压上限")
            obj=json.loads(z.read(item))
        if not 599<=obj["now"]<=1320 or len(obj["records"])>5000:
            raise ValueError("运行包的时钟或订单数量超出范围")
        run=Engine.restore(obj,fork=True)
        self.save(run)
        return run
