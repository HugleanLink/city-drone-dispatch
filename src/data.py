import gzip
import io
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
from .model import BUSINESS, Order, START, LAST_ARRIVAL

KEYS = ["minute", "pickup_latitude", "pickup_longitude", "dropoff_latitude",
        "dropoff_longitude", "biz_line", "package_price"]

@dataclass
class Profile:
    rows: pd.DataFrame
    origin: tuple[float, float]
    report: dict
    single_pool: bool = False


def _minute(values):
    pieces = values.astype("string").str.extract(r"(?:^|[ T])([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?$")
    return pd.to_numeric(pieces[0], errors="coerce")*60 + pd.to_numeric(pieces[1], errors="coerce")


def prepare(source, mapping=None, encoding="utf-8-sig"):
    if isinstance(source, bytes):
        if source[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=io.BytesIO(source)) as archive:
                source=archive.read(200*1024*1024+1)
            if len(source)>200*1024*1024:
                raise ValueError("压缩 CSV 解压后超过 200 MB")
        source=io.BytesIO(source)
    df = pd.read_csv(source, encoding=encoding, dtype="string").rename(columns=mapping or {})
    required = ["customer_pay_time", *KEYS[1:5]]
    if missing := set(required)-set(df):
        raise ValueError("缺少输入列 " + "、".join(sorted(missing)))
    total = len(df)
    df["minute"] = _minute(df.customer_pay_time)
    for key in KEYS[1:5]:
        df[key] = pd.to_numeric(df[key], errors="coerce").astype("float64")
    coords = (df.pickup_latitude.between(-90, 90) & df.dropoff_latitude.between(-90, 90)
              & df.pickup_longitude.between(-180, 180) & df.dropoff_longitude.between(-180, 180))
    df["biz_line"] = df.get("biz_line", pd.Series("waimai", index=df.index)).fillna("waimai")
    supported = df.biz_line.isin(BUSINESS)
    price_text = df.get("package_price", pd.Series(pd.NA, index=df.index))
    price = pd.to_numeric(price_text, errors="coerce").astype("float64")
    price_ok = price_text.isna() | (np.isfinite(price) & (price >= 0))
    df["package_price"] = price
    if "dt_seq_id" in df:
        df["group"] = df.dt_seq_id.fillna("1").astype(str)
    else:
        df["group"] = df.customer_pay_time.str.extract(r"^(\d{4}-\d\d-\d\d)")[0].fillna("1")
    valid = coords & supported & df.minute.notna() & price_ok
    duplicated = pd.Series(False, index=df.index)
    if "order_seq_id" in df:
        has_id = df.order_seq_id.notna() & valid
        rows = df.loc[has_id]
        repeats = rows.duplicated(["group", "order_seq_id"], keep=False)
        if repeats.any():
            check = rows.loc[repeats, ["group", "order_seq_id", *KEYS]].drop_duplicates()
            if check.duplicated(["group", "order_seq_id"]).any():
                raise ValueError("同一日组和订单编号存在内容冲突")
            duplicated.loc[has_id] = rows.duplicated(["group", "order_seq_id"]).to_numpy()
    df["reference_minutes"] = np.nan
    df["weight_kg"] = pd.to_numeric(df.get("weight_kg", pd.Series(np.nan, index=df.index)), errors="coerce").astype(float)
    if (df.weight_kg.notna() & (~np.isfinite(df.weight_kg) | (df.weight_kg <= 0))).any():
        raise ValueError("weight_kg 须为正数")
    reference_late = reference_invalid = 0
    if {"fetch_time","arrived_time"}.issubset(df):
        f, a = _minute(df.fetch_time), _minute(df.arrived_time)
        pre, drive = (f-df.minute)%1440, (a-f)%1440
        df["reference_minutes"] = drive.where((pre + drive) <= 180).astype(float)
        reference_late = int((valid & ((pre + drive)>180)).sum())
        reference_invalid = int((valid & (f.isna() | a.isna())).sum())
    active = valid & ~duplicated & df.minute.between(START, LAST_ARRIVAL-1)
    if not active.any():
        raise ValueError("10:00 至 20:00 内没有可用订单")
    grouping = ["group", *KEYS, "reference_minutes", "weight_kg"]
    rows = df.loc[active, grouping].groupby(
        grouping, sort=True, dropna=False, observed=True
    ).size().rename("count").reset_index()
    lat = pd.concat([rows.pickup_latitude, rows.dropoff_latitude])
    lon = pd.concat([rows.pickup_longitude, rows.dropoff_longitude])
    origin = ((lat.min()+lat.max())/2, (lon.min()+lon.max())/2)
    if lat.max()-lat.min() > .5 or lon.max()-lon.min() > .5 or abs(origin[0]) >= 85:
        raise ValueError("请选择同一模拟城区内的坐标范围")
    report = {"输入": total, "坐标异常": int((~coords).sum()), "时间异常": int(df.minute.isna().sum()),
              "业务异常": int((~supported).sum()), "价格异常": int((~price_ok).sum()),
              "重复": int(duplicated.sum()), "参考时长超过180分钟": reference_late,
              "参考时刻缺失或异常": reference_invalid, "窗口内可用": int(active.sum()),
              "按日组": {str(k):int(v) for k,v in df.loc[active].group.value_counts().items()}}
    return Profile(rows, tuple(float(x) for x in origin), report)


def rng(seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))


def new_order(id, minute, pickup, dropoff, business, random, price=None, source="generated"):
    _, low, high, deadline, kg_min, kg_max = BUSINESS[business]
    return Order(id, minute, minute+int(random.integers(low, high+1)), minute+deadline,
                 tuple(pickup), tuple(dropoff), business, round(float(random.uniform(kg_min, kg_max)), 2),
                 price, source=source)


def day_groups(profile):
    return sorted(profile.rows.group.unique(), key=lambda group: (not group.isdecimal(), int(group) if group.isdecimal() else group))


def profile_for_groups(profile, groups):
    return replace(profile, rows=profile.rows.loc[profile.rows.group.isin(groups)], single_pool=True)


def sources(profile):
    groups = day_groups(profile)
    if len(groups) >= 7 and not profile.single_pool:
        return {"planning": groups[:4], "operation": groups[4:7]}
    return {"planning": groups, "operation": groups}


def source_counts(profile):
    pools = sources(profile)
    return {key: int(profile.rows.loc[profile.rows.group.isin(groups), "count"].sum())
            for key, groups in pools.items()}


def split_sources(profile, planning, operation):
    total = int(profile.rows["count"].sum())
    if not (planning >= 1 and operation >= 1 and planning + operation <= total):
        raise ValueError("规划与运营来源都需要至少一单，合计不能超过有效订单数")
    counts = profile.rows["count"].to_numpy(dtype=np.int64)
    positions = rng(101, 22).permutation(np.repeat(np.arange(len(counts)), counts))

    def pool(indices):
        frequencies = np.bincount(indices, minlength=len(counts))
        selected = frequencies > 0
        return replace(profile, rows=profile.rows.loc[selected].assign(count=frequencies[selected]), single_pool=True)

    return pool(positions[:planning]), pool(positions[planning:planning + operation])


def planning_sample(profile, count=None, seed=101):
    rows = profile.rows[profile.rows.group.isin(sources(profile)["planning"])]
    positions = np.repeat(np.arange(len(rows)), rows["count"].to_numpy(dtype=np.int64))
    if count is not None:
        if not 0 <= count <= len(positions):
            raise ValueError("规划订单数超过来源数量或小于零")
        if count < len(positions):
            positions = rng(seed, 0).choice(positions, size=count, replace=False)
    return materialize(rows.iloc[positions], seed, "P", 1)


def generate(profile, count, seed, purpose="operation", window=None, prefix=None):
    if count < 0 or (purpose != "planning" and count > 100_000) or seed < 0:
        raise ValueError("订单数量或种子超出允许范围")
    pools = sources(profile)
    split = pools["planning"] != pools["operation"]
    if purpose == "planning" and split:
        base = count // len(pools["planning"])
        orders = []
        for i, group in enumerate(pools["planning"]):
            size = base + (i < count % len(pools["planning"]))
            orders.extend(sample(profile.rows[profile.rows.group.eq(group)], size, seed+i, f"P{i+1}"))
        return orders
    rows = profile.rows[profile.rows.group.isin(pools["operation"])] if split else profile.rows
    if window:
        rows = rows[rows.minute.ge(window[0]) & rows.minute.lt(window[1])]
    if purpose == "heavy" and count >= 201 and not window:
        return sample(rows,201,seed,"O")+sample(rows,count-201,seed,"X",4)
    return sample(rows, count, seed, prefix or ("P" if purpose == "planning" else "O"))


def sample(rows, count, seed, prefix, stream=0):
    if count == 0:
        return []
    if rows.empty:
        raise ValueError("所选时窗没有可以抽样的记录")
    weights = rows["count"].to_numpy(dtype=float)
    positions = rng(seed, stream).choice(len(rows), size=count, p=weights/weights.sum())
    return materialize(rows.iloc[positions], seed, prefix, stream + 1)


def materialize(rows, seed, prefix, stream):
    random = rng(seed, stream)
    orders = []
    for i, r in enumerate(rows.itertuples(index=False)):
        order = new_order(f"{prefix}-{seed}-{i+1:05}", int(r.minute),
                      (r.pickup_latitude,r.pickup_longitude), (r.dropoff_latitude,r.dropoff_longitude),
                      r.biz_line, random, None if pd.isna(r.package_price) else float(r.package_price))
        reference = getattr(r, "reference_minutes", np.nan)
        weight = getattr(r, "weight_kg", np.nan)
        orders.append(replace(order, kg=order.kg if pd.isna(weight) else float(weight),
                              reference_minutes=None if pd.isna(reference) else float(reference)))
    return orders
