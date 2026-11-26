from __future__ import annotations

import hashlib
import os
from math import isfinite
import tempfile
from base64 import b64encode
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from .batch import digest
from .speed import fit_hourly
from .data import day_groups, planning_sample, prepare, profile_for_groups
from .model import BUSINESS, MODES, Order
from .workspace import ManualRecord, ManualStore
from .cli import profile_extent
from .deployment import deploy
from .operations import MAX_ORDERS, Dispatcher, grounds_used, joint, operating_cost, saved_schedule, staff_estimate

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
    s.default_staff = staff_estimate(operation_profile)
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
        hubs = st.number_input("贪心 Hub 数量上限", 1, 50, 15)
        radius = st.number_input("两端直线接驳半径 km", min_value=.1, value=3., step=.1)
        weight = st.number_input("无人机最大载重 kg", min_value=.1, value=2., step=.1)
    return dict(max_hubs=hubs, radius=radius, weight=weight)


def data_panel(settings):
    s = st.session_state
    st.subheader("导入与建网")
    upload = st.file_uploader("导入历史运营数据", type=["csv", "gz"], key="data_file")
    if not upload:
        if not s.network:
            st.info("上传 CSV 或 CSV.GZ 后，选择用于规划与估时的历史订单。")
        return
    if s.get("upload_token") != upload.file_id:
        s.upload_token = upload.file_id
        s.upload_error = None
        try:
            with st.spinner("校验历史订单…"):
                payload = upload.getvalue()
                fingerprint = hashlib.sha256(payload).hexdigest()
                try:
                    profile = prepare(payload)
                except UnicodeDecodeError:
                    profile = prepare(payload, encoding="gb18030")
                s.profile, s.profile_key = profile, fingerprint
        except DATA_ERRORS as error:
            s.upload_error = f"导入失败 · {error}"
    if s.upload_error:
        st.error(s.upload_error)
        return
    profile = s.profile
    groups = day_groups(profile)
    selected = st.multiselect("历史日组", groups, default=groups, key=f"history_{s.profile_key}")
    available = sum(profile.report["按日组"][day] for day in selected)
    use_all = st.checkbox("使用所选日组的全部订单", key=f"full_{s.profile_key}")
    count = (None if use_all else st.number_input("建网抽样上限", 1, max(1, available),
                                                 min(4_000, max(1, available)), key=f"count_{s.profile_key}_{available}"))
    if st.button("分析并构建航网", type="primary", disabled=not selected):
        try:
            with st.spinner("拟合距离与分时估时，搜索航网部署方案…"):
                pool = profile_for_groups(profile, selected)
                orders = planning_sample(pool, count)
                network = deploy(orders, pool.origin, profile_extent(pool), **settings)
                ground_model = fit_hourly(pool)
            activate(network, s.profile_key, upload.name[:255], pool, ground_model)
            st.success(f"已启用新航网 · {len(network.routes)} 条可调度路径")
        except DATA_ERRORS as error:
            st.error(f"建网失败 · {error}")


def map_panel(defaults):
    s = st.session_state
    network = s.network
    st.subheader("航网地图")
    diag = network.diagnostics or {}
    corridors = diag.get("hub_corridors", [])
    hubs = len({site["id"] for site in network.candidates if site["active"] and site["role"] == "Hub"}
               | {route.hub for route in network.routes})
    spokes = len({route.spoke for route in network.routes}
                 | {site["id"] for site in network.candidates if site["active"] and site["role"] == "Spoke"})
    st.caption(f"{hubs} 个 Hub、{spokes} 个 Spoke、{len(network.routes)} 条航段")
    if diag.get("distance_window"):
        low, high = diag["distance_window"]
        st.caption(f"适飞距离 {low:.2f}–{high:.2f} km · {diag.get('candidate_method', '未形成候选站点')}"
                   + (" · 使用备用距离区间" if diag["distance_fallback"] else ""))
        with st.expander("航网目标与地面估时"):
            if objectives := diag.get("deployment_objectives"):
                cost, lost, ground = objectives
                st.write(f"综合部署方案 · 样本成本 {cost:,.0f} 元 · "
                         f"价值加权损失率 {lost:.1%} · 地面依赖度 {ground:.1%}")
            st.dataframe([{"出发时段": f"{i:02d}:00", "P50（分钟/km）": round(p50, 2),
                           "P90（分钟/km）": round(p90, 2), "GMM 分量": detail["components"],
                           "拟合样本": detail["samples"]}
                          for i, ((p50, p90), detail) in enumerate(zip(s.ground_model.quantiles,
                                                                     s.ground_model.diagnostics))],
                         hide_index=True)
    if not network.routes:
        st.info("规划订单未形成满足载重、航程与两端接驳条件的航段，地面配送仍可计算。")
    show_corridors = bool(corridors) and st.checkbox(
        "显示 Hub 间规划走廊",
        help="紫色虚线为已启用的 Hub 间走廊。红线为调度可用的 Hub–Spoke 路径。")
    if not network.routes and diag.get("screened_out"):
        st.caption("初筛排除 · " + "；".join(f"{reason}：{count:,} 单" for reason, count in diag["screened_out"].items()))
    target = st.radio("地图点击设置", ("取货点", "目的地"), horizontal=True)
    prefix = f"point_{s.map_revision}_"
    base_map = folium.Map(location=network.origin, zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    left, right, bottom, top = network.bounds
    base_map.fit_bounds([network.latlon(left, bottom), network.latlon(right, top)])
    layers = folium.FeatureGroup(name="航网与新单")
    sites = {}
    for corridor in corridors if show_corridors else ():
        folium.PolyLine([corridor["start"], corridor["end"]], color="#725296", weight=2, dash_array="5 6",
                        tooltip=f"{corridor['id']} · Hub 间规划走廊 · {corridor['km']:.2f} km").add_to(layers)
    for route in network.routes:
        folium.PolyLine([route.pickup, *route.via, route.dropoff], color="#c7212a",
                        weight=2, opacity=.42, tooltip=f"{route.id} · {route.km:.2f} km").add_to(layers)
        sites[route.hub] = (route.pickup, "Hub")
        sites[route.spoke] = (route.dropoff, "Spoke")
    for site in network.candidates:
        if site["active"]:
            sites.setdefault(site["id"], (site["location"], site["role"]))
    for name, (point, role) in sites.items():
        outline = "#394352"
        tooltip = f"{role} {name}"
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


def guide_for(job, ground, air, chosen):
    midnight = job.arrival - job.order.arrival
    eta = lambda plan: plan.eta - midnight if plan else None
    return {"mode": chosen.mode if chosen else "wait", "ground_eta": eta(ground), "air_eta": eta(air),
            "preferred_route_id": job.order.route_id, "air_route_id": air.route if air else None,
            "ready": job.ready - midnight}


def output_rows(jobs, plans, routes):
    rows = []
    for job, plan in zip(jobs, plans):
        midnight = job.arrival - job.order.arrival
        rows.append({"订单": job.order.id[:12], "下单": job.ordered_at.replace("T", " "),
                     "方式": MODES[plan.mode] if plan else "待分配",
                     "航段": plan.route if plan else None, "预计送达": hhmm(plan.eta - midnight) if plan else "—",
                     "端到端（分钟）": plan.eta - job.arrival if plan else None,
                     "骑手占用（分钟）": grounds_used(plan) if plan else None,
                     "预计成本（元）": round(operating_cost(plan, routes), 2) if plan else None})
    return rows


def render_reservations(jobs, plans):
    with st.expander("配送与资源明细"):
        for job, plan in zip(jobs, plans):
            if not plan:
                continue
            midnight = job.arrival - job.order.arrival
            st.write(job.order.id[:12])
            if plan.air_times:
                hub, launch, parcel, handoff = plan.air_times
                st.caption(f"Hub 交接 {hhmm(hub-midnight)} · 起飞 {hhmm(launch-midnight)} · "
                           f"Spoke 可取 {hhmm(parcel-midnight)} · 末段取货 {hhmm(handoff-midnight)}")
            st.dataframe([{"资源": slot.resource, "任务": slot.activity, "开始": hhmm(slot.start-midnight),
                           "结束": hhmm(slot.end-midnight)} for slot in plan.slots], hide_index=True)


def order_panel(defaults, riders, fleet):
    s = st.session_state
    st.subheader("订单调度")
    revision = f"{s.map_revision}_{s.get('order_revision', 0)}"
    number = st.number_input("本次录入订单数", 1, MAX_ORDERS, 1, key=f"count_{revision}")
    day = st.date_input("下单日期", today(), key=f"day_{revision}")
    threshold = st.number_input("单笔采用空地的最小节时（分钟）", 0, 60, 0, key=f"threshold_{revision}")
    fields = ("取货纬度", "取货经度", "送达纬度", "送达经度")
    prefix = f"point_{s.map_revision}_"
    points = [s.get(prefix + name + str(k), fallback[k])
              for name, fallback in zip(("pickup", "dropoff"), defaults) for k in range(2)]
    businesses = {info[0]: code for code, info in BUSINESS.items()}
    if number == 1:
        when = st.time_input("下单时间", time(12), step=60, key=f"time_{revision}")
        with st.expander("取送坐标与订单参数"):
            cols = st.columns(2)
            for i, (label, point) in enumerate(zip(fields, points)):
                key = prefix + ("pickup", "dropoff")[i // 2] + str(i % 2)
                s.setdefault(key, point)
                limit = 90. if i % 2 == 0 else 180.
                points[i] = cols[i // 2].number_input(label, -limit, limit, format="%.7f", key=key)
            business = st.selectbox("业务类型", list(businesses), key=f"business_{revision}")
            kg = st.number_input("包裹重量 kg", min_value=.01, value=1., step=.1, key=f"kg_{revision}")
            price = st.number_input("订单价格（元）", min_value=0., value=None, step=.1, key=f"price_{revision}")
        inputs = [{"下单时间": when.strftime("%H:%M"), **dict(zip(fields, points)),
                   "业务": business, "重量 kg": kg, "价格": price}]
    else:
        base = {"下单时间": "12:00", **dict(zip(fields, points)), "业务": "外卖", "重量 kg": 1., "价格": None}
        frame = pd.DataFrame([base] * number).astype({"价格": "Float64"})
        edited = st.data_editor(frame, key=f"orders_{revision}_{number}", hide_index=True,
                                column_config={"业务": st.column_config.SelectboxColumn(options=list(businesses), required=True),
                                               "重量 kg": st.column_config.NumberColumn(min_value=.01, required=True),
                                               "价格": st.column_config.NumberColumn(min_value=0.),
                                               **{field: st.column_config.NumberColumn(format="%.7f", required=True)
                                                  for field in fields}})
        inputs = edited.astype(object).where(pd.notna(edited), None).to_dict("records")
    input_key = digest([day.isoformat(), inputs, s.network_version])
    signature = digest([input_key, riders, fleet, threshold])
    draft = s.get("draft")
    same = draft and draft["signature"] == signature
    recorded = bool(draft and draft.get("input_key") == input_key and draft["confirmed"])
    retry = recorded and bool(draft["pending"])
    if st.button("重算剩余订单" if retry else "计算配送建议" if number == 1 else "计算联合调度",
                 type="primary", disabled=recorded and not retry):
        try:
            dispatcher = Dispatcher(s.network, s.ground_model, riders, fleet)
            records, version = s.manual_store.confirmed_state(s.source, s.network_version)
            state = saved_schedule(records)
            if retry:
                jobs = draft["pending"]
            else:
                jobs = []
                for item in inputs:
                    when = datetime.strptime(item["下单时间"], "%H:%M").time()
                    stamp = datetime.combine(day, when).isoformat(timespec="minutes")
                    coordinates = [float(item[field]) for field in fields]
                    weight = float(item["重量 kg"])
                    price = item["价格"]
                    if not all(isfinite(x) for x in coordinates) or not isfinite(weight) or weight <= 0:
                        raise ValueError("请填写有效坐标和正数重量")
                    if price is not None and (not isfinite(float(price)) or float(price) < 0):
                        raise ValueError("订单价格须为非负数")
                    minute = clock(when)
                    order = Order("M-" + uuid4().hex, minute, minute + 8, None, tuple(coordinates[:2]),
                                  tuple(coordinates[2:]), businesses[item["业务"]], weight, price, source="manual")
                    jobs.append(dispatcher.job(order, stamp))
            now = max(max(job.arrival for job in jobs), draft["now"] if retry else 0)
            with st.spinner("按当前骑手和无人机状态计算…"):
                if len(jobs) == 1:
                    ground, air, plan = dispatcher.immediate(jobs[0], state, now, threshold)
                    plans, report = (plan,), None
                else:
                    jobs, plans, report = joint(dispatcher, jobs, state, now, threshold)
                    ground = air = None
            s.draft = draft = dict(signature=signature, input_key=input_key, now=now,
                                   jobs=tuple(jobs), plans=plans, ground=ground, air=air,
                                   report=report, version=version, confirmed=False, pending=(),
                                   routes=dispatcher.routes)
            same = True
        except DATA_ERRORS as error:
            st.error(f"计算失败 · {error}")
            return
    if not same and not recorded:
        return
    jobs, plans = draft["jobs"], draft["plans"]
    if draft["confirmed"]:
        if draft["pending"]:
            st.info(f"已确认可执行方案，{len(draft['pending'])} 笔待分配，可重新计算。")
        else:
            st.success("已确认配送计划，可在下方登记送达。")
            if st.button("录入下一批"):
                s.draft = None
                s.order_revision = s.get("order_revision", 0) + 1
                st.rerun()
        return
    if draft["report"]:
        report = draft["report"]
        if report["fallback"]:
            st.info("未找到完整可行的联合方案，按单笔规则分配可执行订单。")
        else:
            weighted, occupied, cost = report["objectives"]
            st.caption(f"{report['pareto']} 个非支配方案 · 价值加权耗时 {weighted:.1f} · "
                       f"骑手占用 {occupied:.0f} 分钟 · 配送成本 {cost:.2f} 元")
    else:
        ground, air = draft["ground"], draft["air"]
        guide = guide_for(jobs[0], ground, air, plans[0])
        cols = st.columns(2)
        cols[0].metric("地面 ETA", hhmm(guide["ground_eta"]))
        cols[1].metric("空地 ETA", hhmm(guide["air_eta"]))
        if ground and air:
            st.caption(f"空地预计节时 {ground.eta-air.eta:+d} 分钟 · {air.route}")
        elif not air:
            st.caption(jobs[0].order.eligibility if jobs[0].order.ground_only else "当前航空运力无法执行")
        st.write(f"配送建议 · {MODES[guide['mode']]}")
        options = {plan.mode: plan for plan in (ground, air) if plan}
        if len(options) > 1:
            modes = list(options)
            chosen = st.radio("人工确认方式", modes, index=modes.index(guide["mode"]),
                              format_func=MODES.get, horizontal=True, key=f"manual_mode_{revision}_{signature[:10]}")
            plans = (options[chosen],)
    st.dataframe(output_rows(jobs, plans, draft["routes"]), hide_index=True)
    render_reservations(jobs, plans)
    if st.button("确认可执行配送方案", disabled=not any(plans)):
        try:
            records = []
            for job, plan in zip(jobs, plans):
                if not plan:
                    continue
                ground, air = ((draft["ground"], draft["air"]) if not draft["report"] else
                               (plan if plan.mode == "ground" else None, plan if plan.mode == "air" else None))
                guide = guide_for(job, ground, air, draft["plans"][0] if not draft["report"] else plan)
                records.append(ManualRecord(job.order, s.source, s.network_version, job.ordered_at,
                                            guide, guide, None, plan.mode, chosen_route_id=plan.route,
                                            reservation=asdict(plan)))
            s.manual_store.confirm_plans(records, s.source, s.network_version, draft["version"])
            draft["pending"] = tuple(job for job, plan in zip(jobs, plans) if not plan)
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
                       "空间首选": row.result.get("preferred_route_id"), "确认航段": row.chosen_route_id,
                       "地面 ETA": hhmm(row.result["ground_eta"]), "空地 ETA": hhmm(row.result["air_eta"]),
                       "实际送达": row.delivered_at.replace("T", " "), "航网版本": row.network_version[:8]}
                      for row in complete[:100]]
            st.dataframe(recent, hide_index=True, width="stretch")
        elif not pending:
            st.caption("确认配送方式后，可以在这里登记实际送达。")


def resource_panel():
    s = st.session_state
    hubs = sorted({site["id"] for site in s.network.candidates if site["active"] and site["role"] == "Hub"}
                  | {route.hub for route in s.network.routes})
    slots = ("00:00–08:00", "08:00–16:00", "16:00–24:00")
    with st.sidebar.expander("运力配置"):
        riders = tuple(st.number_input(f"{label} 骑手数", min_value=0, value=count, step=1,
                                       key=f"riders_{s.map_revision}_{i}")
                       for i, (label, count) in enumerate(zip(slots, s.default_staff)))
        fleet = {hub: st.number_input(f"{hub} 无人机数", 0, 100, 5, key=f"fleet_{s.map_revision}_{hub}")
                 for hub in hubs}
    return riders, fleet


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
    data_panel(planning_settings())
    s = st.session_state
    network = s.network
    if network is None:
        return
    riders, fleet = resource_panel()
    defaults = ((network.routes[0].pickup, network.routes[0].dropoff) if network.routes
                else (network.origin, network.origin))
    st.divider()
    map_column, order_column = st.columns([1.55, 1], gap="large")
    with map_column:
        map_panel(defaults)
    with order_column:
        order_panel(defaults, riders, fleet)
    manual_history()
