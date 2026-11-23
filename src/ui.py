from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event
from uuid import uuid4

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from .batch import DISTURBANCES, SCENARIOS, Disturbance, digest, load_results, run_batch, summarize_results
from .data import new_order, prepare, rng, source_counts
from .demo_data import PUBLIC_COUNT
from .examples import mechanism_network
from .model import BUSINESS, MODES, STRATEGIES, Network
from .storage import encode
from .workspace import DEMO_SOURCE, SAMPLE_SOURCE, ManualRecord, ManualStore, construct, manual_preview

ROOT = Path(__file__).resolve().parents[1]
CLOUD = os.environ.get("DRONE_DISPATCH_MODE") != "local"
TITLE = "城市物流无人机航网规划与空地协同调度系统"
DEMO_LABEL = "公开模拟 CSV"
# Counts of the fixed public file, shown before the full profile is loaded.
DEMO_COUNTS = (338_121, 254_546)
DATA_ERRORS = (ValueError, TypeError, OSError, KeyError, OverflowError, EOFError)
STYLE = """
<style>
.block-container { max-width: 1510px; padding-top: 2.6rem; padding-bottom: 3rem; }
.app-title { color: #122338; font-weight: 750; font-size: clamp(1.5rem, 3vw, 2.45rem);
             letter-spacing: -.045em; line-height: 1.18; margin-bottom: .4rem; }
.app-subtitle { color: #52677e; margin-bottom: 1.3rem; font-size: .95rem; }
.decision { background: white; border-left: 4px solid #0e766e; border-radius: 8px;
            padding: .65rem .9rem; margin: .4rem 0 .65rem; }
.decision small { color: #4e6379; }
.decision strong { display: block; font-size: 1.25rem; color: #122338; }
div[data-testid=stMetric] { background: white; padding: .55rem .7rem; }
[data-testid=stMetricValue], [data-testid=stMetricValue] > div { font-size: 1.28rem; }
</style>
"""


@st.cache_resource(show_spinner=False)
def public_profile():
    return prepare(ROOT / "data/demo_orders.csv.gz")


@st.cache_data(show_spinner=False)
def public_network():
    return Network.from_dict(json.loads((ROOT / "data/demo_network.json").read_text(encoding="utf-8")))


@st.cache_resource
def worker_pool():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="dispatch")


def clock(value):
    return value.hour * 60 + value.minute


def hhmm(value):
    return "不可用" if value is None else f"{value // 60:02}:{value % 60:02}"


def today():
    return datetime.now(timezone(timedelta(hours=8))).date()


def read_source(source):
    s = st.session_state
    if source == DEMO_SOURCE:
        return public_profile()
    if s.profile_key == source and s.profile is not None:
        return s.profile
    raise ValueError("请重新导入该航网对应的源 CSV")


def activate(network, source, label):
    s = st.session_state
    s.network = network
    s.network_version = digest(asdict(network))
    s.source = source
    s.source_label = label
    s.disturbance = None
    s.draft = None
    s.map_click = None
    s.map_revision = s.get("map_revision", 0) + 1


def initialize():
    s = st.session_state
    if "manual_store" not in s:
        if CLOUD:
            s.temp = tempfile.TemporaryDirectory(prefix="dispatch-")
            directory = Path(s.temp.name)
        else:
            directory = Path(os.environ.get("DRONE_DISPATCH_RUNS", ROOT / "runs"))
        s.manual_store = ManualStore(directory / "manual-orders.sqlite3")
        s.profile = s.profile_key = None
        s.build_source = DEMO_SOURCE
        s.build_label = DEMO_LABEL
        activate(public_network(), DEMO_SOURCE, DEMO_LABEL)


def data_panel():
    s = st.session_state
    st.subheader("导入与建网")
    left, right = st.columns([1.65, 1])
    with left:
        upload = st.file_uploader("导入运行数据", type=["csv", "gz"], key=f"data_file_{s.get('upload_revision', 0)}")
        with st.expander("字段映射与编码"):
            mapping = st.text_area("原始列名到标准列名的 JSON 映射", "{}")
            encoding = st.selectbox("CSV 编码", ("utf-8-sig", "gb18030"))
        if upload:
            token = (upload.file_id, mapping, encoding)
            if s.get("upload_token") != token:
                s.upload_token = token
                s.upload_error = None
                try:
                    mapping_data = json.loads(mapping)
                    if not isinstance(mapping_data, dict) or not all(isinstance(v, str) for v in mapping_data.values()):
                        raise ValueError("字段映射必须是列名到列名的 JSON 对象")
                    with st.spinner("校验订单和准备需求分布…"):
                        payload = upload.getvalue()
                        profile = prepare(payload, mapping_data, encoding)
                        fingerprint = hashlib.sha256(payload).hexdigest()
                    s.profile, s.profile_key = profile, fingerprint
                    s.profile_counts = source_counts(profile)
                    s.build_source, s.build_label = fingerprint, upload.name[:255]
                except DATA_ERRORS as error:
                    s.upload_error = f"导入失败 · {error}"
            if s.get("upload_error"):
                st.error(s.upload_error)
        if s.build_source != DEMO_SOURCE and st.button("改用公开模拟 CSV", key="use_demo"):
            s.build_source, s.build_label = DEMO_SOURCE, DEMO_LABEL
            s.upload_revision = s.get("upload_revision", 0) + 1
            st.rerun()
    with right:
        if s.build_source == DEMO_SOURCE:
            total, planning, operation = PUBLIC_COUNT, *DEMO_COUNTS
        elif s.profile_key == s.build_source and s.profile is not None:
            pools = s.profile_counts
            total, planning, operation = s.profile.report["窗口内可用"], pools["planning"], pools["operation"]
        else:
            total = planning = operation = None
        if total is None:
            st.info("已恢复航网。重新导入该工作区的源 CSV 可建网或运行策略对照。")
        else:
            st.caption(f"{s.build_label} · 有效 {total:,} 条 · 规划来源 {planning:,} 条 · 运营来源 {operation:,} 条")
        count = st.number_input("规划订单数", min_value=1, max_value=100_000, value=4_000, step=1)
        include = st.checkbox("纳入已完成的新单", value=True, key="include_manual")
        invalid = bool(upload and s.get("upload_error"))
        if st.button("分析并构建航网", type="primary", disabled=total is None or invalid, width="stretch"):
            try:
                with st.spinner(f"正在分析 {count:,} 条规划订单、选站和构建航段…"):
                    profile = read_source(s.build_source)
                    network = construct(profile, s.manual_store.list(), s.build_source, count, include)
                activate(network, s.build_source, s.build_label)
                st.success(f"已启用新航网 · {len(network.routes)} 条航段 · 纳入 {network.diagnostics['returned_orders']} 条已完成新单")
            except DATA_ERRORS as error:
                st.error(f"建网失败 · {error}")
        if st.button("载入电池扰动样例", key="load_sample"):
            activate(mechanism_network(), SAMPLE_SOURCE, "电池扰动样例")
            s.disturbance = Disturbance("battery", "H01", 50, 720, 780)
            st.rerun()


def station_labels(network):
    labels = {}
    for route in network.routes:
        labels[route.hub_site or route.hub] = route.hub
        labels[route.spoke_site or route.spoke] = route.spoke
    return labels


def disturbance_panel():
    s = st.session_state
    with st.sidebar:
        st.header("扰动调整")
        options = [None, *DISTURBANCES]
        current = s.disturbance
        kind = st.selectbox("调整对象", options, index=options.index(current.kind) if current else 0,
                            format_func=lambda k: DISTURBANCES[k] if k else "无扰动",
                            key=f"disturbance_kind_{s.map_revision}")
        if kind is None:
            if current and st.button("恢复无扰状态", width="stretch"):
                s.disturbance = None
        else:
            labels = station_labels(s.network)
            sites = ([""] if kind == "demand" else sorted(labels) if kind == "pad"
                     else sorted({route.hub for route in s.network.routes}))
            if not sites:
                st.info("当前航网没有可调整的站点。")
            else:
                with st.form(f"disturbance_{s.map_revision}_{kind}"):
                    same = current if current and current.kind == kind else None
                    if kind == "demand":
                        site = ""
                        st.caption("订单量调整用于下方的运营对照。")
                    else:
                        site = st.selectbox("目标站点", sites, index=sites.index(same.site) if same and same.site in sites else 0,
                                            format_func=lambda value: labels.get(value, value))
                    limits = (-100, 200) if kind == "demand" else (1, 100)
                    percent = st.number_input("订单量变化 (%)" if kind == "demand" else "停用比例 (%)",
                                              *limits, same.percent if same else 50, step=1)
                    start_time = time(same.start // 60, same.start % 60) if same else time(12)
                    last = min(same.end, 1320) if same else 740
                    start = st.time_input("开始时间", start_time, step=60)
                    end = st.time_input("结束时间", time(last // 60, last % 60), step=60)
                    if st.form_submit_button("应用扰动", type="primary", width="stretch"):
                        try:
                            s.disturbance = Disturbance(kind, site, percent, clock(start), clock(end))
                        except ValueError as error:
                            st.error(str(error))
        if s.disturbance:
            d = s.disturbance
            location = station_labels(s.network).get(d.site, d.site)
            st.success(f"已应用 {DISTURBANCES[d.kind]} {location} · {d.percent}% · {hhmm(d.start)}–{hhmm(d.end)}")
            if kind and st.button("清除扰动", width="stretch"):
                s.disturbance = None
                st.rerun()
        else:
            st.caption("当前无扰动。")


def map_panel(defaults):
    s = st.session_state
    network = s.network
    st.subheader("航网地图")
    diag = network.diagnostics or {}
    stats = f"{len(network.routes)} 条航段 · {s.source_label}"
    if "planning" in diag:
        stats += f" · {diag['planning']:,} 单规划"
    st.caption(stats)
    if not network.routes:
        st.info("规划订单未形成满足载重、航程与两端接驳条件的航段，地面配送仍可计算。")
    if s.source == SAMPLE_SOURCE:
        st.caption("电池样例含三笔背景订单。默认新单为 12:00，备用电池由 6 块限为 3 块。")
    target = st.radio("地图点击设置", ("取货点", "目的地"), horizontal=True)
    prefix = f"point_{s.map_revision}_"
    d = s.disturbance
    base_map = folium.Map(location=network.origin, zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    left, right, bottom, top = network.bounds
    base_map.fit_bounds([network.latlon(left, bottom), network.latlon(right, top)])
    layers = folium.FeatureGroup(name="航网与新单")
    is_resource = d and d.kind != "demand"
    sites = {}
    for route in network.routes:
        endpoints = {route.hub_site or route.hub, route.spoke_site or route.spoke}
        affected = is_resource and (d.site in endpoints if d.kind == "pad" else d.site == route.hub)
        folium.PolyLine([route.pickup, route.dropoff], color="#c46a27" if affected else "#0e766e",
                        weight=5, dash_array="8 6" if affected else None,
                        tooltip=f"{route.id} · {route.km:.2f} km" + (" · 运力受限" if affected else "")).add_to(layers)
        sites[route.hub] = (route.pickup, "Hub", is_resource and (d.site in (route.hub_site, route.hub)))
        sites[route.spoke] = (route.dropoff, "接驳站", is_resource and d.site in (route.spoke_site, route.spoke))
    for name, (point, role, affected) in sites.items():
        folium.CircleMarker(point, radius=8 if role == "Hub" else 6,
                            color="#c46a27" if affected else "#163555" if role == "Hub" else "#0e766e",
                            fill=True, fill_opacity=.9, tooltip=f"{role} {name}" + (" · 受限" if affected else "")).add_to(layers)
    for index, (key, label, color) in enumerate((("pickup", "取货点", "blue"), ("dropoff", "目的地", "orange"))):
        point = tuple(s.get(prefix + key + str(i), defaults[index][i]) for i in range(2))
        folium.Marker(point, tooltip=label, icon=folium.Icon(color=color,
                      icon="shopping-bag" if key == "pickup" else "flag", prefix="fa")).add_to(layers)
    clicked = st_folium(base_map, key=f"network_map_{s.map_revision}", feature_group_to_add=layers,
                        height=500, use_container_width=True, returned_objects=["last_clicked"]).get("last_clicked")
    if clicked and clicked != s.map_click:
        s.map_click = clicked
        key = "pickup" if target == "取货点" else "dropoff"
        s[prefix + key + "0"], s[prefix + key + "1"] = clicked["lat"], clicked["lng"]
        st.rerun()


def result_panel(draft):
    normal, result = draft["normal"], draft["result"]
    mode = result["mode"]
    st.markdown(f'<div class="decision"><small>推荐方式</small><strong>{MODES[mode]}</strong></div>',
                unsafe_allow_html=True)
    for col, label, key in zip(st.columns(3), ("地面直送", "空地接驳", "交付截止"),
                              ("ground_eta", "air_eta", "deadline")):
        col.metric(label, hhmm(result[key]), border=True)
    st.caption(f"地面 · {result['ground_reason']}　｜　空地 · {result['air_reason']}")
    branches = {item["mode"]: item for item in result["branches"]}
    winner = branches.get(mode)
    other = branches.get({"ground": "air", "air": "ground"}.get(mode))
    if mode == "wait":
        st.write("等待复核，复核时可重新评估资源和交付时限。")
    elif other and winner["on_time"] != other["on_time"]:
        st.write(f"该方式预计准时 {winner['on_time']} 单，另一方式预计准时 {other['on_time']} 单。")
    elif other and winner["ground_on_time"] != other["ground_on_time"]:
        st.write("两种方式预计准时总数相同，推荐方式能多保障仅地面订单。")
    elif other and winner["eta"] < other["eta"]:
        st.write("两种方式的预计准时表现相同，推荐方式预计更早送达。")
    elif other:
        st.write("两种方式的预计准时表现和送达时间相同，采用地面直送。")
    else:
        st.write("该方式能排入当前资源，另一方式无法在备妥时启动。")
    if draft["disturbance"] and draft["disturbance"].kind != "demand":
        st.dataframe([{"条件": title, "推荐方式": MODES[guide["mode"]],
                       "地面 ETA": hhmm(guide["ground_eta"]), "空地 ETA": hhmm(guide["air_eta"])}
                      for title, guide in (("无扰动", normal), ("当前扰动", result))],
                     hide_index=True, width="stretch")
    if result["branches"]:
        with st.expander("查看试排依据"):
            st.dataframe([{"分支": MODES[branch["mode"]], "本单 ETA": hhmm(branch["eta"]),
                           "预计准时数": branch["on_time"], "仅地面预计准时数": branch["ground_on_time"]}
                          for branch in result["branches"]], hide_index=True, width="stretch")


def order_panel(defaults):
    s = st.session_state
    st.subheader("新单配送建议")
    prefix = f"point_{s.map_revision}_"
    options = list(BUSINESS)
    date_col, time_col = st.columns(2)
    day = date_col.date_input("下单日期", today(), key=f"day_{s.map_revision}")
    when = time_col.time_input("下单时间", time(12), step=60, key=f"time_{s.map_revision}")
    with st.expander("取送坐标与订单参数"):
        coords = []
        for col, (key, label, point) in zip(st.columns(2), (
                ("pickup", "取货点", defaults[0]), ("dropoff", "目的地", defaults[1]))):
            lat_key, lon_key = prefix + key + "0", prefix + key + "1"
            s.setdefault(lat_key, point[0])
            s.setdefault(lon_key, point[1])
            lat = col.number_input(f"{label}纬度", min_value=-90., max_value=90., format="%.7f", key=lat_key)
            lon = col.number_input(f"{label}经度", min_value=-180., max_value=180., format="%.7f", key=lon_key)
            coords.append((lat, lon))
        business = st.selectbox("业务类型", options, format_func=lambda kind: BUSINESS[kind][0])
        weight = st.number_input("包裹重量 kg（0 为业务默认）", min_value=0., max_value=1000., step=.1)
    st.caption("在地图上选择取货点和目的地，也可填写坐标。")
    ordered_at = datetime.combine(day, when).isoformat(timespec="minutes")
    signature = digest([ordered_at, coords, business, weight, s.source, s.network_version])
    draft = s.draft
    same = draft and draft["signature"] == signature
    requested = st.button("计算配送建议", type="primary", width="stretch",
                          disabled=bool(same and draft["confirmed"]))
    if requested or (same and not draft["confirmed"] and draft["disturbance"] != s.disturbance):
        try:
            sample = s.source == SAMPLE_SOURCE
            order_id = draft["order"].id if same else "M-" + uuid4().hex
            order = new_order(order_id, clock(when), *coords, business, rng(int(signature[:8], 16), 6), source="manual")
            if sample:
                order = replace(order, ready=order.arrival, deadline=order.arrival + 26, kg=1.)
            if weight:
                order = replace(order, kg=weight)
            normal = manual_preview(s.network, order, sample=sample)
            result = manual_preview(s.network, order, s.disturbance, sample=sample) if s.disturbance else normal
            s.draft = draft = dict(signature=signature, order=order, ordered_at=ordered_at,
                                   normal=normal, result=result, disturbance=s.disturbance,
                                   confirmed=bool(same and draft["confirmed"]))
        except DATA_ERRORS as error:
            s.draft = draft = None
            st.error(f"无法计算配送建议 · {error}")
    if draft is None:
        return
    if draft["signature"] != signature:
        st.info("订单输入已调整，重新计算后可确认。")
        return
    result_panel(draft)
    if draft["confirmed"]:
        st.success(f"订单 {draft['order'].id[:12]} 已确认，可在下方登记送达。")
        if st.button("录入下一单"):
            s.draft = None
            st.rerun()
        return
    modes = [mode for mode in ("ground", "air") if draft["result"][f"{mode}_eta"] is not None]
    if not modes:
        st.info("当前两种方式都无法执行，等待资源恢复后可重新计算。")
        return
    recommended = draft["result"]["mode"]
    widget_version = digest([draft["signature"], asdict(s.disturbance) if s.disturbance else None])[:14]
    chosen = st.radio("人工确认方式", modes, index=modes.index(recommended) if recommended in modes else 0,
                      format_func=MODES.get, key=f"manual_mode_{widget_version}", horizontal=True)
    if st.button("确认送单方式", width="stretch"):
        try:
            saved = ManualRecord(draft["order"], s.source, s.network_version, draft["ordered_at"],
                                 draft["normal"], draft["result"],
                                 asdict(draft["disturbance"]) if draft["disturbance"] else None, chosen)
            s.manual_store.confirm(saved)
            draft["confirmed"] = True
            st.rerun()
        except DATA_ERRORS as error:
            st.error(str(error))


def manual_history():
    s = st.session_state
    records = s.manual_store.list()
    pending = {row.order.id: row for row in records if row.delivered_at is None}
    complete = [row for row in records if row.delivered_at is not None]
    with st.expander(f"手工订单 · {len(pending)} 单待送达 · {len(complete)} 单已完成", expanded=bool(pending)):
        if pending:
            order_id = st.selectbox("待送达订单", list(pending), format_func=lambda id:
                                    f"{id[:12]} · {MODES[pending[id].chosen_mode]}")
            record = pending[order_id]
            cols = st.columns(3)
            cols[0].caption(f"数据来源 · {record.source[:12]}　｜　下单 · {record.ordered_at.replace('T', ' ')}")
            default_day = datetime.fromisoformat(record.ordered_at).date()
            day = cols[1].date_input("实际送达日期", default_day, key=f"delivered_day_{order_id}")
            actual = cols[2].time_input("实际送达时间", value=None, step=60, key=f"delivered_time_{order_id}")
            if st.button("登记实际送达", type="primary", disabled=actual is None):
                try:
                    s.manual_store.complete(order_id, datetime.combine(day, actual).isoformat(timespec="minutes"))
                    s.delivery_notice = f"订单 {order_id[:12]} 已完成并入库。"
                    st.rerun()
                except DATA_ERRORS as error:
                    st.error(str(error))
        if notice := s.pop("delivery_notice", None):
            st.success(notice)
        if complete:
            recent = [{"订单": row.order.id[:12], "下单": row.ordered_at.replace("T", " "),
                       "建议": MODES[row.result["mode"]], "人工确认": MODES[row.chosen_mode],
                       "地面 ETA": hhmm(row.result["ground_eta"]), "空地 ETA": hhmm(row.result["air_eta"]),
                       "实际送达": row.delivered_at.replace("T", " "), "航网版本": row.network_version[:8]}
                      for row in complete[:100]]
            st.dataframe(recent, hide_index=True, width="stretch")
        elif not pending:
            st.caption("确认配送方式后，可以在这里登记实际送达。")


def backup_panel():
    s = st.session_state
    with st.expander("工作区备份与恢复"):
        try:
            package = s.manual_store.export(s.network, s.source, s.source_label)
            st.download_button("下载工作区 ZIP", package, "dispatch-workspace.zip", "application/zip",
                               on_click="ignore")
        except DATA_ERRORS as error:
            st.error(str(error))
        backup = st.file_uploader("选择工作区 ZIP", type=["zip"], key="workspace_backup")
        if backup and st.button("恢复工作区"):
            try:
                network, source, label = s.manual_store.restore(backup.getvalue())
                activate(network, source, label)
                if source != SAMPLE_SOURCE:
                    s.build_source, s.build_label = source, label
                    s.upload_revision = s.get("upload_revision", 0) + 1
                else:
                    s.disturbance = Disturbance("battery", "H01", 50, 720, 780)
                s.restore_notice = "已恢复航网和手工订单。"
                st.rerun()
            except (*DATA_ERRORS, zipfile.BadZipFile) as error:
                st.error(f"恢复失败 · {error}")
        if notice := s.pop("restore_notice", None):
            st.success(notice)


def show_comparison(rows, suffix):
    if not rows:
        return
    summary = summarize_results(rows)
    versions = {row["network_sha256"] for row in summary}
    if len(versions) == 1 and next(iter(versions)):
        st.caption(f"结果航网版本 · {next(iter(versions))[:12]}")
    table = []
    for row in summary:
        item = {"场景": SCENARIOS[row["scenario"]], "策略": STRATEGIES[row["strategy"]],
                "已完成日数": row["days"]}
        if len(versions) > 1:
            item["航网版本"] = row["network_sha256"][:12] or "旧版"
        for title, numerator, denominator, metric in (
                ("全体", "on_time", "arrivals", "on_time_rate"),
                ("仅地面", "ground_on_time", "ground_only", "ground_on_time_rate")):
            rate, delta = row[metric], row[f"{metric}_delta"]
            item[f"{title}准时率"] = (f"{rate:.1%}（{row[numerator]:,}/{row[denominator]:,}）"
                                      if rate is not None else "无样本（0/0）")
            item[f"{title}较纯地面"] = f"{delta:+.1f} pp" if delta is not None else "无配对样本"
        table.append(item)
    st.dataframe(table, hide_index=True, width="stretch")
    if any(row.get("input_sha256") for row in rows):
        with st.expander("查看共同输入哈希"):
            st.dataframe([{"模拟日种子": row["seed"], "场景": SCENARIOS[row["scenario"]],
                           "策略": STRATEGIES[row["strategy"]], "输入哈希": row["input_sha256"]}
                          for row in rows], hide_index=True, width="stretch")
    flat = [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in rows]
    first, second = st.columns(2)
    first.download_button("下载对照 CSV", pd.DataFrame(flat).to_csv(index=False).encode("utf-8-sig"),
                          "batch.csv", "text/csv", on_click="ignore", key=f"batch_csv_{suffix}")
    second.download_button("下载完整结果 JSON", encode(rows), "batch.json", "application/json",
                           on_click="ignore", key=f"batch_json_{suffix}")


def batch_panel():
    s = st.session_state
    st.subheader("运营策略对照")
    job = s.get("batch")
    busy = bool(job and not job["future"].done())
    count_col, action = st.columns([1, 1.45], vertical_alignment="bottom")
    count = count_col.number_input("运营订单数／模拟日", 1, 500, 201, step=1)
    ready = (s.source == DEMO_SOURCE or (s.source != SAMPLE_SOURCE and s.profile_key == s.source
                                       and s.profile is not None))
    if action.button("运行策略对照", type="primary", width="stretch", disabled=busy or not ready):
        try:
            with st.spinner("读取运营需求分布…"):
                profile = read_source(s.source)
            queues, cancelled = SimpleQueue(), Event()
            network, disturbance = s.network, s.disturbance
            future = worker_pool().submit(run_batch, profile, network, (201,), ("normal",),
                                          orders_per_day=count, disturbance=disturbance,
                                          cancelled=cancelled.is_set, progress=queues.put)
            s.batch = {"future": future, "queue": queues, "cancelled": cancelled, "rows": [],
                       "count": count, "version": s.network_version, "done": False, "disturbance": disturbance}
            st.rerun()
        except DATA_ERRORS as error:
            st.error(f"无法启动策略对照 · {error}")
    if not ready:
        st.caption("重新导入航网对应的源 CSV 可运行对照。" if s.source != SAMPLE_SOURCE else
                   "电池样例用于单笔演示。重新建网后可运行运营对照。")

    if job:
        @st.fragment(run_every=1.0 if busy else None)
        def progress_panel():
            while True:
                try:
                    job["rows"].append(job["queue"].get_nowait())
                except Empty:
                    break
            changed = job["disturbance"]
            condition = f" · {DISTURBANCES[changed.kind]} {changed.percent}%" if changed else ""
            st.caption(f"本次输入 · {job['count']} 单 · 种子 201 · 航网 {job['version'][:12]}{condition}")
            count_done = len(job["rows"])
            st.progress(count_done / len(STRATEGIES), text=f"策略已完成 {count_done}／{len(STRATEGIES)}")
            if job["future"].done():
                if busy and not job["done"]:
                    job["done"] = True
                    st.rerun()
                if error := job["future"].exception():
                    st.error(f"策略对照失败 · {error}")
                elif job["cancelled"].is_set() and count_done < len(STRATEGIES):
                    st.info("已取消。下方显示本次已完成的策略。")
                if count_done < len(STRATEGIES):
                    missing = set(STRATEGIES) - {row["strategy"] for row in job["rows"]}
                    st.caption("未完成 · " + "、".join(label for key, label in STRATEGIES.items() if key in missing))
                show_comparison(job["rows"], "current")
            elif job["cancelled"].is_set():
                st.caption("正在停止当前策略…")
            elif st.button("取消策略对照"):
                job["cancelled"].set()
        progress_panel()

    with st.expander("导入多日对照结果"):
        upload = st.file_uploader("选择命令行生成的 batch.json 或 batch.csv", type=["json", "csv"],
                                  key="batch_file")
        if upload and st.button("读取对照结果"):
            try:
                rows = load_results(upload.getvalue(), upload.name.lower().endswith(".csv"))
                s.imported_batch = (upload.name, rows)
            except DATA_ERRORS as error:
                st.error(f"读取对照结果失败 · {error}")
        if s.get("imported_batch"):
            name, rows = s.imported_batch
            st.caption(f"{name} · {len(rows)} 条策略记录")
            complete = {}
            for row in rows:
                group = (row["network_sha256"], row["scenario"], row["seed"], row["input_sha256"])
                complete.setdefault(group, set()).add(row["strategy"])
            missing = sum(len(group) < len(STRATEGIES) for group in complete.values())
            if missing:
                st.caption(f"{missing} 个配对组的策略尚未全部完成。")
            show_comparison(rows, "imported")


def main():
    st.set_page_config(page_title=TITLE, layout="wide", page_icon="✈️")
    st.markdown(STYLE, unsafe_allow_html=True)
    initialize()
    st.markdown(f'<div class="app-title">{TITLE}</div>'
                '<div class="app-subtitle">Drone Delivery Network and Air–Ground Dispatch</div>',
                unsafe_allow_html=True)
    data_panel()
    disturbance_panel()

    s = st.session_state
    network = s.network
    defaults = ((network.routes[0].pickup, network.routes[0].dropoff) if network.routes
                else (network.origin, network.origin))
    st.divider()
    map_column, order_column = st.columns([1.55, 1], gap="large")
    with map_column:
        map_panel(defaults)
    with order_column:
        order_panel(defaults)
    manual_history()
    st.divider()
    batch_panel()
    backup_panel()
