from __future__ import annotations

import hashlib
import os
import tempfile
from base64 import b64encode
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

from .batch import DISTURBANCES, Disturbance, adjust_demand, digest, run_batch
from .calibration import fit
from .efficiency import fleet_estimate, operating_settings
from .data import day_groups, generate, new_order, prepare, profile_for_groups, rng, source_counts, split_sources
from .model import BUSINESS, MODES, STRATEGIES, FlightRules
from .storage import encode
from .workspace import ManualRecord, ManualStore, construct, manual_preview

ROOT = Path(__file__).resolve().parents[1]
CLOUD = os.environ.get("DRONE_DISPATCH_MODE") != "local"
TITLE = "城市物流无人机航网规划与空地协同调度系统"
DATA_ERRORS = (ValueError, TypeError, OSError, KeyError, OverflowError, EOFError)
STYLE = """
<style>
.block-container { max-width: 1510px; padding-top: 2.6rem; padding-bottom: 3rem; }
.brand-title { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center;
               column-gap: .85rem; row-gap: .3rem; margin-bottom: 1.3rem; }
.brand-title img { display: block; width: clamp(38px, 4.5vw, 60px); height: auto; }
.brand-title .app-title { color: #122338; font-weight: 750; font-size: clamp(1.5rem, 3vw, 2.45rem);
                          letter-spacing: -.045em; line-height: 1.18; margin: 0; padding: 0; }
.app-subtitle { grid-column: 2; color: #52677e; font-size: .95rem; }
.decision { background: white; border-left: 4px solid #0e766e; border-radius: 8px;
            padding: .65rem .9rem; margin: .4rem 0 .65rem; }
.decision small { color: #4e6379; }
.decision strong { display: block; font-size: 1.25rem; color: #122338; }
.eta-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .45rem; }
.eta-cell { padding: .6rem .48rem; border: 1px solid #d8e1e9; border-radius: .5rem; background: white; }
.eta-cell span { display: block; font-size: .8rem; color: #52677e; }
.eta-cell strong { display: block; color: #122338; font-size: clamp(.9rem, 1.8vw, 1.35rem);
                   font-variant-numeric: tabular-nums; line-height: 1.35; }
</style>
"""


@st.cache_resource
def worker_pool():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="dispatch")


def clock(value):
    return value.hour * 60 + value.minute


def hhmm(value):
    if value is None:
        return "不可用"
    day, minute = divmod(value, 1440)
    prefix = "次日 " if day == 1 else f"第 {day+1} 日 " if day else ""
    return f"{prefix}{minute // 60:02}:{minute % 60:02}"


def today():
    return datetime.now(timezone(timedelta(hours=8))).date()


def activate(network, source, label, operation_profile, ground_model):
    s = st.session_state
    s.network = network
    s.operation_profile = operation_profile
    s.ground_model = ground_model
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
        s.profile = s.profile_key = s.network = None
        s.operation_profile = None


def planning_settings():
    with st.sidebar.expander("建网参数"):
        weight = st.number_input("包裹最大重量 kg", min_value=.01, value=2., step=.1)
        order_min = st.number_input("最短取送直线距离 km", min_value=0., value=1., step=.1)
        order_max = st.number_input("最长取送直线距离 km", min_value=.01, value=4.04, step=.1)
        access = st.number_input("每端最大直线接驳距离 km", min_value=.01, value=3., step=.1)
        leg_min = st.number_input("最短 Hub–接驳站航段 km", min_value=.01, value=.5, step=.1)
        leg_max = st.number_input("最长 Hub–接驳站航段 km", min_value=.01, value=6.06, step=.1)
        hubs = st.number_input("Hub 数量上限（0 为不限）", min_value=0, value=15, step=1)
        spokes = st.number_input("接驳站数量上限（0 为不限）", min_value=0, value=10, step=1)
        corridors = st.number_input("可调度航段数量上限（0 为不限）", min_value=0, value=0, step=1)
        eps = st.number_input("DBSCAN 邻域半径 km（0 为自动）", min_value=0., value=0., step=.05)
        minimum = st.number_input("DBSCAN 最小样本数（0 为自动）", min_value=0, value=0, step=1)
        exclude = st.checkbox("拼好饭仅走地面", value=True)
        st.caption("订单距离与接驳半径按直线计算。CSV 没有重量时，使用按业务类型生成的重量。站点候选来自上传 CSV 的取送点。")
    rules = dict(weight=weight, order_min=order_min, order_max=order_max, access_km=access,
                 leg_min=leg_min, leg_max=leg_max, report=True, exclude_group_meal=exclude)
    settings = dict(max_hubs=hubs or None, max_spokes=spokes or None, max_routes=corridors or None,
                    eps=eps or None, minimum=minimum or None)
    return rules, settings


def day_ranges(group_count, source):
    start_key, end_key, op_start_key, op_end_key = (f"{source}_{part}" for part in
                                                   ("plan_start", "plan_end", "op_start", "op_end"))
    state = st.session_state

    def set_operation_range():
        if state[end_key] < group_count:
            state[op_start_key], state[op_end_key] = state[end_key] + 1, group_count
        elif state[start_key] > 1:
            state[op_start_key], state[op_end_key] = 1, state[start_key] - 1

    def range_fields(label, first_key, last_key, first, last, on_change=None):
        title, start, separator, end, unit = st.columns([2, 1, .25, 1, .4], vertical_alignment="center")
        title.write(label)
        separator.write("–")
        unit.write("日")
        lower = start.number_input(label + "起始日", 1, group_count, first, key=first_key,
                                   label_visibility="collapsed", on_change=on_change)
        upper = end.number_input(label + "结束日", 1, group_count, last, key=last_key,
                                 label_visibility="collapsed", on_change=on_change)
        return lower, upper

    default_end = max(1, min(4, group_count - 1))
    plan = range_fields("规划航网使用", start_key, end_key, 1, default_end, set_operation_range)
    operation = range_fields("模拟运营使用", op_start_key, op_end_key, min(default_end + 1, group_count), group_count)
    return plan, operation


def data_panel(rules, settings):
    s = st.session_state
    st.subheader("导入与建网")
    upload = st.file_uploader("导入运行数据", type=["csv", "gz"], key="data_file")
    if not upload:
        s.upload_token = None
        s.profile = s.profile_key = None
        if not s.network:
            st.info("上传 CSV 或 CSV.GZ 后可划分数据并构建航网。")
        return
    if s.get("upload_token") != upload.file_id:
        s.upload_token = upload.file_id
        s.upload_error = None
        try:
            with st.spinner("校验 CSV 并统计有效订单…"):
                payload = upload.getvalue()
                try:
                    profile = prepare(payload)
                except UnicodeDecodeError:
                    profile = prepare(payload, encoding="gb18030")
                fingerprint = hashlib.sha256(payload).hexdigest()
            s.profile, s.profile_key = profile, fingerprint
            s.profile_counts = source_counts(profile)
        except DATA_ERRORS as error:
            s.upload_error = f"导入失败 · {error}"
    if s.upload_error:
        st.error(s.upload_error)
        return
    profile = s.profile
    total = profile.report["窗口内可用"]
    planning, operation = s.profile_counts["planning"], s.profile_counts["operation"]
    manual_split = False
    if total >= 2:
        split = st.radio("数据来源划分", ("按日组划分", "手动指定来源数量"), horizontal=True)
        manual_split = split == "手动指定来源数量"
    if manual_split:
        first, second = st.columns(2)
        planning = first.number_input("规划来源数量", 1, total - 1, min(planning, total - 1),
                                      key=f"planning_source_{s.profile_key}")
        operation = second.number_input("运营来源数量", 1, total - 1, min(operation, total - planning),
                                        key=f"operation_source_{s.profile_key}")
        valid = planning + operation <= total
        if valid:
            st.caption(f"两组不重叠，有 {total - planning - operation:,} 条不使用")
        else:
            st.error("规划与运营来源数量之和不能超过有效订单数。")
    else:
        groups = day_groups(profile)
        (plan_first, plan_last), (op_first, op_last) = day_ranges(len(groups), s.profile_key)
        valid = plan_first <= plan_last and op_first <= op_last
        if not valid:
            st.error("起始日须早于或等于结束日。")
        elif len(groups) > 1 and not (plan_last < op_first or op_last < plan_first):
            valid = False
            st.error("规划和运营的日期范围不能重叠。")
        plan_days, operation_days = groups[plan_first-1:plan_last], groups[op_first-1:op_last]
        planning = sum(profile.report["按日组"][day] for day in plan_days)
        if len(groups) == 1:
            st.caption("文件只有一个有效日组。可用手动指定来源数量来拆分同日订单。")
    use_all = st.checkbox("使用全部规划来源")
    count = None if use_all else st.number_input("规划订单数", min_value=1, value=max(1, min(4_000, planning)),
                                                key=f"planning_count_{s.profile_key}_{planning}")
    if valid and count is not None and count > planning:
        st.error("规划订单数不能超过规划来源数量。")
        valid = False
    include = st.checkbox("纳入已完成的新单", value=True, key="include_manual")
    if st.button("分析并构建航网", type="primary", disabled=not valid):
        try:
            with st.spinner(f"正在分析 {planning if count is None else count:,} 条规划订单并构建航网…"):
                if manual_split:
                    plan_pool, operation_pool = split_sources(profile, planning, operation)
                else:
                    plan_pool = profile_for_groups(profile, plan_days)
                    operation_pool = profile_for_groups(profile, operation_days)
                ground_model = fit(plan_pool)
                network = construct(plan_pool, s.manual_store.list(), s.profile_key, count, include,
                                    no_replacement=True, rules=FlightRules(**rules), **settings)
            activate(network, s.profile_key, upload.name[:255], operation_pool, ground_model)
            returned = network.diagnostics["returned_orders"]
            st.success(f"已启用新航网 · {len(network.routes)} 条可调度航段 · 纳入 {returned} 条已完成新单")
        except DATA_ERRORS as error:
            st.error(f"建网失败 · {error}")


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
                    last = same.end % 1440 if same else 740
                    start = st.time_input("开始时间", start_time, step=60)
                    end = st.time_input("结束时间", time(last // 60, last % 60), step=60)
                    if st.form_submit_button("应用扰动", type="primary", width="stretch"):
                        try:
                            begin, finish = clock(start), clock(end)
                            if finish < begin:
                                finish += 1440
                            s.disturbance = Disturbance(kind, site, percent, begin, finish)
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
    corridors = diag.get("hub_corridors", [])
    hubs = len({site["id"] for site in network.candidates if site["active"] and site["role"] == "Hub"}
               | {route.hub for route in network.routes})
    spokes = len({route.spoke for route in network.routes})
    st.caption(f"{hubs} 个 Hub、{spokes} 个 Spoke、{len(network.routes)} 条航段")
    if not network.routes:
        st.info("规划订单未形成满足载重、航程与两端接驳条件的航段，地面配送仍可计算。")
    show_corridors = bool(corridors) and st.checkbox(
        "显示 Hub 间规划走廊",
        help="紫色虚线连接两个 Hub，供规划时查看中转通道。红线连接 Hub 和 Spoke，用于当前订单的空地接驳调度。")
    if not network.routes and diag.get("screened_out"):
        st.caption("初筛排除 · " + "；".join(f"{reason}：{count:,} 单" for reason, count in diag["screened_out"].items()))
    target = st.radio("地图点击设置", ("取货点", "目的地"), horizontal=True)
    prefix = f"point_{s.map_revision}_"
    d = s.disturbance
    base_map = folium.Map(location=network.origin, zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    left, right, bottom, top = network.bounds
    base_map.fit_bounds([network.latlon(left, bottom), network.latlon(right, top)])
    layers = folium.FeatureGroup(name="航网与新单")
    is_resource = d and d.kind != "demand"
    sites = {}
    for corridor in corridors if show_corridors else ():
        folium.PolyLine([corridor["start"], corridor["end"]], color="#725296", weight=2, dash_array="5 6",
                        tooltip=f"{corridor['id']} · Hub 间规划走廊 · {corridor['km']:.2f} km").add_to(layers)
    for route in network.routes:
        endpoints = {route.hub_site or route.hub, route.spoke_site or route.spoke}
        affected = is_resource and (d.site in endpoints if d.kind == "pad" else d.site == route.hub)
        folium.PolyLine([route.pickup, route.dropoff], color="#c7212a",
                        weight=2, opacity=.68 if affected else .42, dash_array="6 4" if affected else None,
                        tooltip=f"{route.id} · {route.km:.2f} km" + (" · 运力受限" if affected else "")).add_to(layers)
        sites[route.hub] = (route.pickup, "Hub", is_resource and (d.site in (route.hub_site, route.hub)))
        sites[route.spoke] = (route.dropoff, "接驳站", is_resource and d.site in (route.spoke_site, route.spoke))
    for site in network.candidates:
        if site["active"] and site["role"] == "Hub":
            sites.setdefault(site["id"], (site["location"], "Hub", False))
    for name, (point, role, affected) in sites.items():
        outline = "#c46a27" if affected else "#394352"
        tooltip = f"{role} {name}" + (" · 受限" if affected else "")
        if role == "Hub":
            marker = folium.CircleMarker(point, radius=7, color=outline, weight=2,
                                         fill=True, fill_color="#ffdb44", fill_opacity=1, tooltip=tooltip)
        else:
            marker = folium.RegularPolygonMarker(point, number_of_sides=3, radius=5, rotation=0,
                                                  color=outline, weight=1, fill_color="#fff",
                                                  fill_opacity=1, tooltip=tooltip)
        marker.add_to(layers)
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
    cards = "".join(f'<div class="eta-cell"><span>{label}</span><strong>{hhmm(result[key])}</strong></div>'
                    for label, key in (("地面直送", "ground_eta"), ("空地接驳", "air_eta"), ("预计备妥", "ready")))
    st.markdown(f'<div class="eta-strip">{cards}</div>', unsafe_allow_html=True)
    st.caption(f"地面 · {result['ground_reason']}　｜　空地 · {result['air_reason']}")
    branches = {item["mode"]: item for item in result["branches"]}
    winner = branches.get(mode)
    other = branches.get({"ground": "air", "air": "ground"}.get(mode))
    if mode == "wait":
        st.write("等待复核。后续有资源可用时重新评估。")
    elif other and winner["total_minutes"] < other["total_minutes"]:
        st.write("推荐方式的预计配送总耗时更短。")
    else:
        st.write("推荐方式可按当前共享运力安排。")
    if draft["disturbance"] and draft["disturbance"].kind != "demand":
        st.dataframe([{"条件": title, "推荐方式": MODES[guide["mode"]],
                       "地面 ETA": hhmm(guide["ground_eta"]), "空地 ETA": hhmm(guide["air_eta"])}
                      for title, guide in (("无扰动", normal), ("当前扰动", result))],
                     hide_index=True, width="stretch")
    if result["branches"]:
        with st.expander("查看试排依据"):
            st.dataframe([{"分支": MODES[branch["mode"]], "本单 ETA": hhmm(branch["eta"]),
                           "预计完成数": branch["completed"], "预计总耗时（分钟）": branch["total_minutes"]}
                          for branch in result["branches"]], hide_index=True, width="stretch")


def order_panel(defaults, settings):
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
    signature = digest([ordered_at, coords, business, weight, s.source, s.network_version, settings, s.ground_model.version])
    draft = s.draft
    same = draft and draft["signature"] == signature
    requested = st.button("计算配送建议", type="primary", width="stretch",
                          disabled=bool(same and draft["confirmed"]))
    if requested or (same and not draft["confirmed"] and draft["disturbance"] != s.disturbance):
        try:
            order_id = draft["order"].id if same else "M-" + uuid4().hex
            order = new_order(order_id, clock(when), *coords, business, rng(int(signature[:8], 16), 6), source="manual")
            if weight:
                order = replace(order, kg=weight)
            ready = s.ground_model.ready(order)
            order = replace(order, ready=ready)
            options = dict(ground_model=s.ground_model, settings=settings)
            normal = manual_preview(s.network, order, **options)
            result = (manual_preview(s.network, order, s.disturbance, **options)
                      if s.disturbance and s.disturbance.kind != "demand" else normal)
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
    with st.expander("手工订单", expanded=bool(pending)):
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


def resource_panel(count):
    s = st.session_state
    key = (s.map_revision, s.network_version, count)
    if s.get("auto_resources_key") != key:
        automatic = operating_settings(s.operation_profile, s.network, count)
        orders = generate(s.operation_profile, count, 201)
        automatic["aircraft_by_hub"] = fleet_estimate(s.network, orders)
        s.auto_resources, s.auto_resources_key = automatic, key
    settings = dict(s.auto_resources)
    with st.sidebar.expander("运力配置"):
        st.caption(f"默认高峰在线 {max(settings['rider_hours'])} 名骑手，"
                   f"全网 {sum(settings['aircraft_by_hub'].values())} 架无人机。")
        manual = st.checkbox("手动设置骑手和无人机")
        hours = pd.DataFrame({"时段": [f"{hour:02d}:00" for hour in range(24)], "在线骑手": settings["rider_hours"]})
        fleet = pd.DataFrame({"Hub": list(settings["aircraft_by_hub"]),
                              "无人机": list(settings["aircraft_by_hub"].values())})
        editor_key = digest(key)[:12]
        if manual:
            hours = st.data_editor(hours, key=f"rider_hours_{editor_key}", hide_index=True,
                                   disabled=["时段"], column_config={"在线骑手": st.column_config.NumberColumn(
                                       required=True, min_value=0, max_value=500, step=1)})
            if not fleet.empty:
                fleet = st.data_editor(fleet, key=f"fleet_{editor_key}", hide_index=True, disabled=["Hub"],
                                       column_config={"无人机": st.column_config.NumberColumn(
                                           required=True, min_value=0, max_value=100, step=1)})
        else:
            st.dataframe(hours, hide_index=True, height=180)
            if not fleet.empty:
                st.dataframe(fleet, hide_index=True, height=155)
        settings["rider_hours"] = hours["在线骑手"].astype(int).tolist()
        settings["aircraft_by_hub"] = {hub: int(count) for hub, count in zip(fleet.Hub, fleet["无人机"])}
        if manual:
            st.caption(f"当前高峰在线 {max(settings['rider_hours'])} 人，共 {sum(settings['aircraft_by_hub'].values())} 架无人机。")
        for field, label, default in (("battery", "每 Hub 备用电池", 6), ("pads", "每站起降位", 1),
                                       ("swap", "每 Hub 换电位", 1), ("charge", "每 Hub 充电位", 2)):
            settings[field] = st.number_input(label, 0, 100, default, step=1)
    return settings


def show_comparison(rows, suffix):
    if not rows:
        return

    def formatted(value, pattern):
        return format(value, pattern) if value is not None else "无配对样本"

    table = []
    for row in rows:
        table.append({"策略": STRATEGIES[row["strategy"]], "已送达／到达": f"{row['completed']}／{row['arrivals']}",
                      "完成率": formatted(row["completion_rate"], ".1%"), "未送达": row["undelivered"],
                      "配对单数": row["paired_orders"], "单均节省（分钟）": formatted(row["saved_minutes"], "+.1f"),
                      "耗时降幅": formatted(row["time_reduction"], "+.1%"),
                      "仅地面配对数": row["paired_ground"],
                      "仅地面少等（分钟）": formatted(row["saved_ground_wait"], "+.1f"),
                      "仅地面 P90 缩短（分钟）": formatted(row["saved_ground_p90"], "+.1f"),
                      "净节省骑手分钟": row["saved_rider_minutes"]})
    st.caption("与纯地面逐单配对。时长只计两侧均已送达的订单。节省为负表示耗时或占用增加。")
    st.dataframe(table, hide_index=True, width="stretch")
    breakdown = pd.DataFrame([{"策略": STRATEGIES[row["strategy"]], "时段": hhmm(item["hour"]*60),
                               "区域": f"骑手初始区域 {item['zone']}", "任务": item["activity"],
                               "骑手分钟": item["minutes"]}
                              for row in rows for item in row["rider_breakdown"]])
    if not breakdown.empty:
        st.markdown("**骑手占用**")
        choice = st.radio("查看维度", ["按策略", "按时段和区域"], horizontal=True)
        if choice == "按时段和区域":
            names = list(breakdown["策略"].unique())
            strategy, region = st.columns(2)
            name = strategy.selectbox("策略", names)
            zone = region.selectbox("骑手初始区域", ["全部", *sorted(breakdown["区域"].unique())])
            breakdown = breakdown[breakdown["策略"].eq(name)]
            if zone != "全部":
                breakdown = breakdown[breakdown["区域"].eq(zone)]
        dimension = "策略" if choice == "按策略" else "时段"
        chart = breakdown.groupby([dimension, "任务"], as_index=False)["骑手分钟"].sum()
        st.bar_chart(chart, x=dimension, y="骑手分钟", color="任务", stack=True)

    case_row = next((row for row in reversed(rows) if row.get("example")), None)
    with st.expander("查看骑手时间重新分配案例"):
        if case_row:
            case = case_row["example"]
            st.write(f"{STRATEGIES[case_row['strategy']]}中，可飞单 {case['flyable']} 改为空地接驳。"
                     f"骑手 {case['rider']} 在 {hhmm(case['start'])} 至 {hhmm(case['end'])} "
                     f"为仅地面单 {case['ground_only']} 服务，该单提前 {case['earlier_start']} 分钟派发、"
                     f"提前 {case['earlier_delivery']} 分钟送达。")
        else:
            st.caption("本轮已完成策略中没有找到可核对的同骑手转用案例。")

    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    first, second = st.columns(2)
    first.download_button("下载对照 CSV", pd.DataFrame(flat).to_csv(index=False).encode("utf-8-sig"),
                          "batch.csv", "text/csv", on_click="ignore", key=f"batch_csv_{suffix}")
    second.download_button("下载完整结果 JSON", encode(rows), "batch.json", "application/json",
                           on_click="ignore", key=f"batch_json_{suffix}")


def batch_panel(settings):
    s = st.session_state
    st.subheader("配送效率对照")
    job = s.get("batch")
    busy = bool(job and not job["future"].done())
    count_col, action = st.columns([1, 1.45], vertical_alignment="bottom")
    count = count_col.number_input("本轮仿真订单数", 1, 500, 201, step=1, key="efficiency_count")
    ready = s.operation_profile is not None
    if action.button("运行配送效率对照", type="primary", width="stretch", disabled=busy or not ready):
        try:
            profile = s.operation_profile
            queues, cancelled = SimpleQueue(), Event()
            network, disturbance = s.network, s.disturbance
            future = worker_pool().submit(run_batch, profile, network, (201,), ("normal",),
                                          orders_per_day=count, disturbance=disturbance, settings=settings,
                                          ground_model=s.ground_model, include_orders=True,
                                          cancelled=cancelled.is_set, progress=queues.put)
            s.batch = {"future": future, "queue": queues, "cancelled": cancelled, "rows": [],
                       "count": count, "version": s.network_version, "done": False, "disturbance": disturbance,
                       "peak_riders": max(settings["rider_hours"]), "fleet": sum(settings["aircraft_by_hub"].values())}
            st.rerun()
        except DATA_ERRORS as error:
            st.error(f"无法启动策略对照 · {error}")
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
            st.caption(f"本次输入 · {job['count']} 单 · 高峰在线 {job['peak_riders']} 名骑手 · "
                       f"{job['fleet']} 架无人机 · 航网 {job['version'][:12]}{condition}")
            count_done = len(job["rows"])
            st.progress(count_done / len(STRATEGIES), text=f"策略已完成 {count_done}／{len(STRATEGIES)}")
            if job["future"].done():
                if busy and not job["done"]:
                    job["done"] = True
                    st.rerun()
                if error := job["future"].exception():
                    st.error(f"配送效率对照失败 · {error}")
                elif job["cancelled"].is_set() and count_done < len(STRATEGIES):
                    st.info("已取消。下方显示本次已完成的策略。")
                if count_done < len(STRATEGIES):
                    missing = set(STRATEGIES) - {row["strategy"] for row in job["rows"]}
                    st.caption("未完成 · " + "、".join(label for key, label in STRATEGIES.items() if key in missing))
                show_comparison(job["rows"], "current")
            elif job["cancelled"].is_set():
                st.caption("正在停止当前策略…")
            elif st.button("取消对照"):
                job["cancelled"].set()
        progress_panel()


@st.cache_data(show_spinner=False)
def header_logo():
    return b64encode((ROOT / "assets" / "cauc.png").read_bytes()).decode("ascii")


def main():
    st.set_page_config(page_title=TITLE, layout="wide", page_icon="✈️")
    st.markdown(STYLE, unsafe_allow_html=True)
    initialize()
    logo = header_logo()
    st.markdown(f'<div class="brand-title"><img src="data:image/png;base64,{logo}" alt="中国民航大学">'
                f'<h1 class="app-title">{TITLE}</h1>'
                '<div class="app-subtitle">Drone Delivery Network and Air–Ground Dispatch</div></div>',
                unsafe_allow_html=True)
    rules, settings = planning_settings()
    data_panel(rules, settings)
    s = st.session_state
    network = s.network
    if network is None:
        return
    disturbance_panel()
    settings = resource_panel(s.get("efficiency_count", 201))
    defaults = ((network.routes[0].pickup, network.routes[0].dropoff) if network.routes
                else (network.origin, network.origin))
    st.divider()
    map_column, order_column = st.columns([1.55, 1], gap="large")
    with map_column:
        map_panel(defaults)
    with order_column:
        order_panel(defaults, settings)
    manual_history()
    st.divider()
    batch_panel(settings)
