from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import time as Time
from hashlib import sha256
from pathlib import Path
from threading import Event
from uuid import uuid4

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from .batch import SCENARIOS, run_batch, target_hub
from .cli import profile_extent
from .data import generate, new_order, prepare, rng
from .examples import battery_pair, three_orders
from .metrics import metrics, paired
from .model import BUSINESS, END, LAST_ARRIVAL, MODES, START, STRATEGIES, Network, Order
from .network import build
from .sim import Engine
from .storage import RunStore, encode

ROOT = Path(__file__).resolve().parents[1]
CLOUD = os.environ.get("DRONE_DISPATCH_MODE") != "local"
TITLE = "城市物流无人机航网规划与空地协同调度系统"
ENGLISH = "Drone Delivery Network and Air–Ground Dispatch"
PAGES = ("新单决策","运行控制","需求与建网","数据与场景","扰动调整","回放与对照")
MODE_COLOR={"ground":"#E39238","air":"#0E766E","wait":"#707A88"}
STATUS={"scheduled":"未到达","unready":"未备妥","waiting":"待派发","assigned":"执行中",
        "delivered":"已交付","incomplete":"未完成","removed":"已移除"}


@st.cache_resource(show_spinner=False)
def public_profile():
    return prepare(ROOT/"data/demo_orders.csv.gz")


@st.cache_resource(show_spinner=False)
def public_scenario():
    with gzip.open(ROOT/"data/demo_planning.json.gz","rt",encoding="utf-8") as file:
        planning=[Order.from_dict(row) for row in json.load(file)]
    saved=Network.from_dict(json.loads((ROOT/"data/demo_network.json").read_text(encoding="utf-8")))
    return planning,saved


@st.cache_resource
def worker_pool():
    return ThreadPoolExecutor(max_workers=1,thread_name_prefix="scenario-batch")


def hhmm(value):
    return "—" if value is None else f"{value//60:02}:{value%60:02}"


def clock(value):
    return value.hour*60+value.minute


def duration(value):
    return "—" if value is None else f"{value:.1f} 分钟"


def pct(value):
    return "—" if value is None else f"{value:.1%}"


def as_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def state():
    return st.session_state


def set_run(run):
    s=state()
    s.store.save(run)
    s.run=run
    s.initial=json.loads(encode(run.snapshot()))
    s.play=False
    s.goal=None
    s.draft=None


def profile():
    s=state()
    if "profile" not in s:
        with st.spinner("准备完整模拟需求…"):
            s.profile=public_profile()
    return s.profile


def network():
    s=state()
    if "network" not in s:
        with st.spinner("生成规划需求和航网…"):
            if "planning" not in s and ("profile" not in s or s.origin_name=="公开模拟 CSV"):
                s.planning,s.network=public_scenario()
            else:
                p=profile()
                if "planning" not in s:
                    s.planning=generate(p,4000,101,"planning")
                s.network=build(s.planning,p.origin,extent=profile_extent(p))
    return s.network


def get_run():
    if state().get("run") is None:
        set_run(Engine(network()))
    return state().run


def save():
    state().store.save(state().run)


def route_frame(net):
    return [{"航段":r.id,"主站":r.hub,"末端站":r.spoke,"单程 km":round(r.km,2),"往返 km":round(2*r.km,2),"可覆盖订单":r.coverage}
            for r in net.routes]


def network_chart(net, *, selected=None, candidate=False, run=None, height=405):
    fig=go.Figure()
    left,right,bottom,top=net.bounds
    for values,vertical in ((range(round(left*2),round(right*2)+1),True),
                            (range(round(bottom*2),round(top*2)+1),False)):
        for i in values:
            x0,x1,y0,y1=(i/2,i/2,bottom,top) if vertical else (left,right,i/2,i/2)
            fig.add_shape(type="line",x0=x0,x1=x1,y0=y0,y1=y1,line=dict(color="#E3EAF2",width=1),layer="below")
    for i,r in enumerate(net.routes):
        ax,ay=net.xy(r.pickup); bx,by=net.xy(r.dropoff)
        fig.add_trace(go.Scatter(x=[ax,bx],y=[ay,by],mode="lines",name="低空航段",legendgroup="routes",
                      showlegend=i==0,customdata=[r.id,r.id],line=dict(color="#0E766E",width=2.5),
                      hovertemplate="%{customdata}<extra></extra>"))
    candidates=net.candidates
    if not candidates:
        candidates=tuple({ "id":id,"location":location,"role":role,"active":True,"count":0 }
                         for id,location,role in dict.fromkeys((id,location,role) for r in net.routes
                         for id,location,role in ((r.hub,r.pickup,"Hub"),(r.spoke,r.dropoff,"Spoke"))))
    for role,color,symbol in (("Hub","#163555","square"),("Spoke","#0E766E","circle")):
        rows=[row for row in candidates if row["role"]==role and (candidate or row["active"])]
        if rows:
            xy=[net.xy(row["location"]) for row in rows]
            fig.add_trace(go.Scatter(x=[p[0] for p in xy],y=[p[1] for p in xy],mode="markers+text",name=role,
                          text=[row["id"] if row["active"] else "" for row in rows],textposition="top center",
                          marker=dict(color=color,size=[11 if r["active"] else 6 for r in rows],
                                      opacity=[1 if r["active"] else .35 for r in rows],symbol=symbol),
                          customdata=[[r["id"],r["count"]] for r in rows],
                          hovertemplate="%{customdata[0]} · %{customdata[1]} 单<extra></extra>"))
    if selected:
        xy=[net.xy(p) for p in selected]
        fig.add_trace(go.Scatter(x=[p[0] for p in xy],y=[p[1] for p in xy],mode="markers+text",
                      name="新订单",text=["取货","目的地"],textposition="bottom center",
                      marker=dict(symbol=["diamond","star"],size=14,color="#E39238")))
    if run:
        coords=[]
        for rider,(end,point) in run.cal.tails.items():
            if end<=run.now: coords.append(net.xy(point))
        if coords:
            fig.add_trace(go.Scatter(x=[p[0] for p in coords],y=[p[1] for p in coords],
                          name="空闲骑手",mode="markers",marker=dict(size=5,color="#8A6598")))
    fig.update_layout(height=height,margin=dict(l=5,r=8,t=13,b=5),plot_bgcolor="#F7FAFD",
                      paper_bgcolor="#FFFFFF",legend=dict(orientation="h",y=1.1),hovermode="closest")
    fig.update_xaxes(title_text="东西向 km",showgrid=False,zeroline=False,range=[left-.5,right+.5])
    fig.update_yaxes(title_text="南北向 km",showgrid=False,zeroline=False,scaleanchor="x",scaleratio=1,range=[bottom-.5,top+.5])
    st.plotly_chart(fig,width="stretch",config={"displaylogo":False})


def geographic(net,key,select=False,selected=None):
    left,right,bottom,top=net.bounds
    m=folium.Map(location=net.origin,zoom_start=13,tiles="OpenStreetMap",control_scale=True)
    m.fit_bounds([net.latlon(left,bottom),net.latlon(right,top)])
    for r in net.routes:
        folium.PolyLine([r.pickup,r.dropoff],color="#0E766E",weight=4,tooltip=r.id).add_to(m)
    sites=dict.fromkeys((id,point) for r in net.routes for id,point in ((r.hub,r.pickup),(r.spoke,r.dropoff)))
    for id,point in sites:
        folium.CircleMarker(point,radius=6,color="#163555",fill=True,tooltip=id).add_to(m)
    for label,point in zip(("取货","目的地"),selected or ()):
        folium.CircleMarker(point,radius=7,color="#E39238",fill=True,tooltip=label).add_to(m)
    return st_folium(m,key=key,height=410,use_container_width=True,
                     returned_objects=["last_clicked"] if select else [])


def metric_strip(run):
    m=metrics(run)
    captions=[("仿真时刻",hhmm(max(run.now,START))),
              ("订单已到达",m["arrivals"]),("待处理",m["pending"]),
              ("已交付准时率",pct(m["delivered_on_time_rate"]))]
    for col,(label,val) in zip(st.columns(4),captions):
        col.metric(label,val,border=True)


def show_decision(result):
    selected=result["mode"]
    st.markdown(f'<div class="recommend" style="border-left-color:{MODE_COLOR[selected]}">'
                f'<span>调度建议</span><strong>{MODES[selected]}</strong></div>',unsafe_allow_html=True)
    for col,label,k in zip(st.columns(3),("地面预计送达","空地预计送达","交付截止"),("ground_eta","air_eta","deadline")):
        col.metric(label,hhmm(result[k]),border=True)
    st.caption(f"空地可行性 · {result['air_reason']}  ｜  地面可行性 · {result.get('ground_reason','见预计送达')}")
    winning=next((x for x in result["branches"] if x["mode"]==selected),None)
    if winning:
        st.write(f"所选分支预计准时 **{winning['on_time']} 单**，其中仅地面订单 **{winning['ground_on_time']} 单**。")
        with st.expander("查看其他分支及受影响订单"):
            st.dataframe([{"分支":MODES[b["mode"]],"本单 ETA":hhmm(b["eta"]),"全体预计准时":b["on_time"],
                           "仅地面预计准时":b["ground_on_time"],
                           "相较推荐方案失去准时机会":"、".join(b["loses_vs_recommendation"]) or "—"}
                          for b in result["branches"]],hide_index=True,width="stretch")


def decision_page():
    s=state()
    run=get_run()
    st.subheader("新订单配送建议")
    metric_strip(run)
    net=run.network
    default=(net.routes[0].pickup,net.routes[0].dropoff) if net.routes else (net.origin,net.origin)
    left,right=st.columns([1.32,1],gap="large")
    with left:
        st.markdown("#### 取送位置与航网")
        use_geo=st.toggle("在线地图",value=False,key="decision_geo")
        if use_geo:
            target=st.radio("点击地图设置",["取货点","目的地"],horizontal=True)
            clicks=geographic(net,"new_geo",select=True,selected=default)
            if (loc:=clicks.get("last_clicked")) and (point:=(loc["lat"],loc["lng"]))!=s.get("map_last"):
                s.map_last=point
                prefix="pickup" if target=="取货点" else "dropoff"
                s[f"{prefix}_lat"],s[f"{prefix}_lon"]=point
        else:
            points=tuple((s.get(f"{key}_lat",loc[0]),s.get(f"{key}_lon",loc[1]))
                         for key,loc in zip(("pickup","dropoff"),default))
            network_chart(net,selected=points)
        if net.routes:
            st.caption(f"{len(net.routes)} 条启用航段 · 可在地图上查看站点和连接")
    with right:
        if run.now>=LAST_ARRIVAL-1:
            st.info("本次运行已过接单窗口。可在运行控制中创建或重置运行。")
        else:
            with st.form("new_order"):
                minute=max(START,run.now+1)
                default_time=Time(minute//60,minute%60)
                selected_time=st.time_input("下单时间",default_time,step=60,key=f"order_clock_{run.run_id}_{minute}")
                a,b=st.columns(2)
                for col,key,label,loc in ((a,"pickup","取货点",default[0]),(b,"dropoff","目的地",default[1])):
                    with col:
                        st.markdown(f"**{label}**")
                        st.number_input(f"{label}纬度",value=float(s.get(f"{key}_lat",loc[0])),format="%.7f",key=f"{key}_lat",label_visibility="collapsed")
                        st.number_input(f"{label}经度",value=float(s.get(f"{key}_lon",loc[1])),format="%.7f",key=f"{key}_lon",label_visibility="collapsed")
                with st.expander("业务与时限"):
                    business=st.selectbox("业务",list(BUSINESS),format_func=lambda x:BUSINESS[x][0])
                    kg=st.number_input("重量 kg · 0 使用业务参数",min_value=0.,max_value=1000.,step=.1)
                    ready=st.number_input("备妥分钟 · -1 使用业务参数",min_value=-1,max_value=300,value=-1)
                    deadline=st.number_input("时限分钟 · 0 使用业务参数",min_value=0,max_value=300,value=0)
                preview=st.form_submit_button("计算配送建议",type="primary",width="stretch")
            if preview:
                from dataclasses import replace
                try:
                    s.play=False
                    payload=(clock(selected_time),(s.pickup_lat,s.pickup_lon),(s.dropoff_lat,s.dropoff_lon),business,kg,ready,deadline)
                    key=sha256(encode(payload).encode()).hexdigest()
                    request=s.draft["request"] if s.draft and s.draft["key"]==key else uuid4().hex
                    order=new_order("M-"+request[:10],payload[0],payload[1],payload[2],business,rng(int(key[:8],16),6),source="manual")
                    order=replace(order,kg=kg or order.kg,ready=order.ready if ready<0 else order.arrival+ready,
                                  deadline=order.arrival+deadline if deadline else order.deadline)
                    result=run.preview(order)
                    s.draft=dict(order=order,result=result,request=request,key=key,run=run.run_id)
                except (ValueError,OverflowError) as err:
                    s.draft=None
                    st.error(str(err))
        draft=s.get("draft")
        if draft and draft["run"]==run.run_id:
            show_decision(draft["result"])
            if draft["request"] in run.requests:
                st.success(f"订单 {run.requests[draft['request']]} 已登记")
            elif st.button("提交订单到当前运行",width="stretch"):
                try:
                    if draft["result"]["version"]!=run.version:
                        draft["result"]=run.preview(draft["order"])
                        st.info("状态已更新，已重新计算，请核对后提交。")
                        st.rerun()
                    id=run.add(draft["order"],draft["request"])
                    save()
                    st.success(f"订单 {id} 已登记")
                except ValueError as err:
                    st.error(str(err))


def run_controls():
    s=state()
    st.subheader("运行控制")
    with st.expander("创建场景或载入样例",expanded=s.get("run") is None):
        left,right=st.columns(2)
        style=left.selectbox("调度策略",list(STRATEGIES),index=3,format_func=STRATEGIES.get)
        preset=right.selectbox("场景",["空队列","常态 201 单","高需求 302 单","三单试排","备用电池 · 原状态","备用电池 · 停用三块"])
        seed=left.number_input("随机种子",min_value=0,value=201,max_value=2**31-1)
        with st.expander("资源配置"):
            c=st.columns(3)
            counts=[c[i%3].number_input(label,min_value=0,max_value=20,value=value,key=f"setting_{key}")
                    for i,(key,label,value) in enumerate([("riders","每区骑手",4),("aircraft","每 Hub 无人机",4),
                       ("battery","每 Hub 备用电池",6),("pads","每站起降位",1),("swap","每 Hub 换电位",1),("charge","每 Hub 充电位",2)])]
        if st.button("创建运行",type="primary"):
            try:
                if preset=="三单试排":
                    run=three_orders(style)
                elif preset.startswith("备用电池"):
                    run=battery_pair("停用" in preset,style)
                else:
                    net=network()
                    orders=[] if preset=="空队列" else generate(profile(),302 if "302" in preset else 201,seed,
                                                                "heavy" if "302" in preset else "operation")
                    run=Engine(net,orders,style,dict(zip(("riders","aircraft","battery","pads","swap","charge"),counts)))
                set_run(run)
            except ValueError as err: st.error(str(err))
    run=get_run()
    controls=st.columns(5)
    if controls[0].button("开始",disabled=run.now>=END,width="stretch"):
        s.play=True
    if controls[1].button("暂停",width="stretch"):
        s.play=False; s.goal=None
    if controls[2].button("下一事件",disabled=run.now>=END,width="stretch"):
        s.play=False; run.advance(run.next_event()); save()
    if controls[3].button("推进 1 分钟",disabled=run.now>=END,width="stretch"):
        s.play=False; run.advance(min(END,run.now+1)); save()
    if controls[4].button("重置运行",width="stretch"):
        # Reuse the recorded, immutable starting snapshot for this browser session.
        set_run(Engine.restore(s.get("initial",run.snapshot()),fork=True))
        run=s.run
    cols=st.columns(3)
    speed=cols[0].selectbox("播放速度", [1,5,20],index=1,format_func=lambda x:f"{x} 倍")
    done=Time(22,0)
    target=cols[1].time_input("暂停时刻",done,step=60)
    if cols[2].button("运行至指定时刻",width="stretch"):
        if not run.now<clock(target)<=END:
            st.warning("暂停时刻须晚于当前时刻且不超过 22:00。")
        else:
            s.goal=clock(target);s.play=True
    @st.fragment(run_every=.35 if s.play else None)
    def monitor():
        run=s.run
        if s.play:
            end=min(END,s.goal or END)
            run.advance(min(end,run.now+speed))
            save()
            if run.now>=end:
                s.play=False
                st.rerun()
        metric_strip(run)
        net,stats=st.columns([1.35,1],gap="large")
        with net: network_chart(run.network,run=run,height=360)
        with stats:
            st.markdown("#### 站点资源")
            resources=[]
            for hub,groups in run.hubs.items():
                for kind,ids in groups.items():
                    if kind=="mounted": continue
                    restricted=sum(bool(run.cal.limits.get(id)) for id in ids)
                    occupied=sum(any(slot.end>run.now for slot in run.cal.slots.get(id,())) for id in ids)
                    free=sum(not run.cal.limits.get(id) and not any(slot.end>run.now for slot in run.cal.slots.get(id,())) for id in ids)
                    resources.append({"站点":hub,"资源":{"aircraft":"无人机","battery":"备用电池","swap":"换电位","charge":"充电位"}[kind],
                                      "配置":len(ids),"承诺/周转":occupied,"受限":restricted,"可调用":free})
            for site,ids in run.pads.items():
                occupied=sum(any(slot.start<=run.now<slot.end for slot in run.cal.slots.get(id,())) for id in ids)
                resources.append({"站点":next((r.hub for r in run.network.routes if r.hub_site==site),
                                                 next((r.spoke for r in run.network.routes if r.spoke_site==site),site)),
                                  "资源":"起降位","配置":len(ids),"承诺/周转":occupied,
                                  "受限":sum(bool(run.cal.limits.get(id)) for id in ids),
                                  "可调用":sum(not run.cal.limits.get(id) and not any(slot.start<=run.now<slot.end for slot in run.cal.slots.get(id,())) for id in ids)})
            st.dataframe(resources,hide_index=True,height=310)
        timeline=[{"资源":slot.resource,"开始":slot.start,"时长":slot.end-slot.start,
                   "订单":slot.order,"活动":slot.activity}
                  for id,slots in run.cal.slots.items() if id.startswith(("pad:","aircraft:"))
                  for slot in slots if slot.end>run.now-60]
        if timeline:
            from plotly import express as px
            fig=px.bar(pd.DataFrame(timeline),x="时长",y="资源",base="开始",color="活动",hover_data=["订单"],orientation="h")
            fig.update_xaxes(title="仿真分钟（从 00:00 计）")
            fig.update_layout(barmode="overlay",height=340,margin=dict(t=10))
            st.markdown("#### 起降与飞行时间轴")
            st.plotly_chart(fig,width="stretch")
        pending=[{"订单":r["order"].id,"状态":STATUS[r["status"]],"截止":hhmm(r["order"].deadline),
                  "意向":MODES.get(r["intent"],"—")}
                 for r in run.records.values() if r["status"] in ("waiting","unready","assigned")]
        with st.expander("当前订单队列",expanded=bool(pending)):
            if pending: st.dataframe(pending,hide_index=True)
            else: st.caption("当前没有待处理订单。")
    monitor()
    c1,c2=st.columns(2)
    if c1.button("保存配对起点",disabled=s.play):
        s.pair=json.loads(encode(run.snapshot()));st.success("已保存起点")
    if c2.button("从起点创建配对运行",disabled="pair" not in s or s.play):
        set_run(Engine.restore(s.pair,fork=True));st.rerun()


def data_page():
    s=state()
    st.subheader("数据与场景")
    st.write("从全量订单建立时空联合分布，再分别生成规划与运营订单。")
    col1,col2=st.columns([1,1])
    with col1:
        if st.button("载入公开模拟 CSV",type="primary"):
            with st.spinner("正在读取 592,667 条模拟订单…"):
                s.profile=public_profile()
                s.origin_name="公开模拟 CSV"
                s.pop("network",None);s.pop("planning",None)
            st.rerun()
        if not CLOUD:
            source=st.text_input("本地 CSV 路径",value=str(next((ROOT.parent/"流程图和实施路线").glob("*.csv"),"")))
        else:
            source=""
        upload=st.file_uploader("上传新的需求数据",type=["csv","gz"])
        with st.expander("字段映射与编码"):
            mapping=st.text_area("原始列名到标准列名的 JSON 映射",value="{}",height=80)
            encoding=st.selectbox("编码",["utf-8-sig","gb18030"])
        if st.button("校验并准备需求分布",disabled=not (source or upload)):
            try:
                with st.spinner("正在校验并汇总…"):
                    s.profile=prepare(upload.getvalue() if upload else Path(source),json.loads(mapping),encoding)
                s.origin_name=upload.name if upload else Path(source).name
                s.pop("network",None);s.pop("planning",None)
                st.rerun()
            except (ValueError,TypeError,OSError) as err:
                st.error(str(err))
    with col2:
        p=profile()
        r=p.report
        st.metric("输入订单",f"{r['输入']:,}")
        st.metric("运营窗口内有效",f"{r['窗口内可用']:,}")
        st.caption(s.origin_name)
        with st.expander("查看校验和分组"):
            st.json(r)
    p=profile()
    charts=st.columns(2)
    with charts[0]:
        st.markdown("#### 下单时段")
        by_time=p.rows.groupby(p.rows.minute//30)["count"].sum()
        st.bar_chart(pd.Series(by_time.to_numpy(),index=[hhmm(int(i)*30) for i in by_time.index]),
                     color="#0E766E",height=200)
    with charts[1]:
        st.markdown("#### 业务构成")
        business=p.rows.groupby("biz_line")["count"].sum()
        st.bar_chart(pd.Series(business.to_numpy(),index=[BUSINESS[i][0] for i in business.index]),
                     color="#274C73",height=200)
    with st.expander("查看取送点分布"):
        from plotly import express as px
        field=st.radio("端点",["pickup","dropoff"],format_func={"pickup":"取货点","dropoff":"送达点"}.get,horizontal=True)
        points=p.rows.groupby([field+"_latitude",field+"_longitude"],as_index=False)["count"].sum()
        points=points.nlargest(1800,"count")
        fig=px.scatter(points,x=field+"_longitude",y=field+"_latitude",size="count",color="count",
                       labels={field+"_longitude":"经度",field+"_latitude":"纬度","count":"订单量"},
                       color_continuous_scale="Teal",height=330)
        st.plotly_chart(fig,width="stretch")
    with st.form("generate_case"):
        c=st.columns(3)
        purpose=c[0].selectbox("订单用途",["planning","operation","heavy"],format_func={"planning":"规划","operation":"运营","heavy":"高需求"}.get)
        count=c[1].number_input("订单数量",1,5000,4000)
        seed=c[2].number_input("随机种子",0,2**31-1,101)
        if st.form_submit_button("生成模拟订单"):
            try:
                generated=generate(p,count,seed,purpose)
                s.generated=generated
                if purpose=="planning":
                    s.planning=generated
                    s.pop("network",None)
                st.success(f"已生成 {len(generated):,} 单")
            except ValueError as err: st.error(str(err))
    if "generated" in s:
        rows=[asdict(o) for o in s.generated]
        st.download_button("下载生成订单",as_csv(rows),"orders.csv","text/csv")
        st.dataframe(rows[:50],hide_index=True)


def network_page():
    s=state()
    st.subheader("需求与建网")
    p=profile()
    cols=st.columns(4)
    count=cols[0].number_input("规划订单数",100,8000,4000,step=100)
    hubs=cols[1].number_input("Hub 上限",1,10,3)
    spokes=cols[2].number_input("Spoke 上限",1,10,4)
    routes=cols[3].number_input("航段上限",1,20,6)
    if st.button("分析需求并生成航网",type="primary"):
        with st.spinner("聚类、评估候选站并选取航段…"):
            planning=s.get("planning")
            if planning is None or len(planning)!=count:
                planning=generate(p,count,101,"planning")
            s.network=build(planning,p.origin,max_hubs=hubs,max_spokes=spokes,max_routes=routes,extent=profile_extent(p))
            s.planning=planning
            s.pop("covered",None)
    net=network()
    d=net.diagnostics or {}
    for col,label,value in zip(st.columns(4),("规划订单","潜在适飞","联网覆盖","启用航段"),
                              (d.get("planning","—"),d.get("potential","—"),d.get("covered","—"),len(net.routes))):
        col.metric(label,value,border=True)
    if d.get("planning"):
        st.caption(f"联网覆盖率 {d.get('covered',0)/d['planning']:.1%} · "
                   f"名义空地用时优于地面 {d.get('nominal_air_faster',0)} 单")
    if not net.routes:
        st.info("当前规划订单未形成满足覆盖和航程条件的航段。该网络可用于纯地面运行。")
    selected = st.radio("显示方式",["坐标示意","在线地图"],horizontal=True)
    show_candidates=st.toggle("显示候选站",value=True)
    if selected=="在线地图":
        geographic(net,"network_geo")
    else:
        network_chart(net,candidate=show_candidates)
    st.dataframe(route_frame(net),hide_index=True,width="stretch")
    if d.get("cluster_flows"):
        with st.expander("取送需求簇及流量"):
            st.write(f"取货需求簇 {d['pickup_clusters']} 个，送达需求簇 {d['dropoff_clusters']} 个。编号 -1 表示离散点。")
            st.dataframe(d["cluster_flows"],hide_index=True,column_config={"pickup":"取货簇","dropoff":"送达簇","orders":"单量"})
            st.download_button("下载起讫簇流量",as_csv(d["cluster_flows"]),"cluster_flows.csv","text/csv")
    if net.routes and "planning" in s:
        from .model import match_order
        rid=st.selectbox("查看结构覆盖订单",["仅地面",*[r.id for r in net.routes]])
        if st.button("列出对应订单"):
            match=[match_order(net,o) for o in s.planning if net.contains(o.pickup) and net.contains(o.dropoff)]
            s.covered=[asdict(o) for o in match if o.route_id==(None if rid=="仅地面" else rid)]
        if "covered" in s:
            st.dataframe(s.covered[:100],hide_index=True)
            st.download_button("下载完整覆盖明细",as_csv(s.covered),"coverage.csv","text/csv")
    a,b=st.columns(2)
    a.download_button("保存航网 JSON",encode(asdict(net)),"network.json","application/json")
    imported=b.file_uploader("载入已保存航网",type="json",key="network_file")
    if imported and b.button("应用航网"):
        try:
            s.network=Network.from_dict(json.loads(imported.getvalue()))
            s.pop("planning",None)
            s.pop("covered",None)
            st.success("航网已载入，可在运行控制中创建运行。")
        except (ValueError,TypeError,KeyError) as e: st.error(f"航网文件无法载入：{e}")


def disruption_page():
    s=state()
    run=get_run()
    st.subheader("扰动调整")
    metric_strip(run)
    if s.play:
        st.info("请先暂停当前运行。")
        return
    minute=max(START,run.now+1)
    if minute>=LAST_ARRIVAL:
        st.info("本次运行已过扰动录入窗口。")
        return
    kind=st.selectbox("调整对象",["battery","aircraft","pad","demand"],
                      format_func={"battery":"满电备用电池","aircraft":"无人机","pad":"起降位","demand":"未来订单"}.get)
    if kind=="demand":
        with st.form("demand"):
            start=st.time_input("时窗开始",Time(minute//60,minute%60),step=60)
            end=st.time_input("时窗结束",Time(20,0),step=60)
            value=st.number_input("目标订单数",0,500,40)
            apply=st.form_submit_button("提交需求调整")
        if apply:
            try:
                first,last=clock(start),clock(end)
                count=sum(first<=r["order"].arrival<last and r["status"]=="scheduled" for r in run.records.values())
                extra=generate(profile(),max(0,value-count),1_000_000+run.seq,window=(first,last),prefix="D")
                run.adjust_orders(first,last,value,extra)
                save();st.success(f"未来订单由 {count} 单调整为 {value} 单")
            except ValueError as e:st.error(str(e))
    else:
        targets=list(run.pads if kind=="pad" else run.hubs)
        if not targets:
            st.info("当前航网没有对应站点。")
            return
        labels={**{(r.hub_site or r.hub):r.hub for r in run.network.routes},
                **{(r.spoke_site or r.spoke):r.spoke for r in run.network.routes}}
        target=st.selectbox("目标站点",targets,format_func=lambda id:labels.get(id,id))
        candidates=run.resources(kind,target)
        affected=st.number_input("停用数量",0,len(candidates),min(3 if kind=="battery" else 1,len(candidates)))
        c=st.columns(2)
        start=c[0].time_input("生效时间",Time(minute//60,minute%60),step=60)
        end=c[1].time_input("恢复时间",Time(20,0),step=60)
        keep=st.checkbox("本次运行持续受限")
        currently=[id for id in candidates if not run.cal.limits.get(id)
                   and (kind!="battery" or not any(slot.end>run.now for slot in run.cal.slots.get(id,())))]
        if st.button("预览拟受影响资源"):
            st.write({"配置数量":len(candidates),"当前可选":len(currently),"拟受影响":sorted(currently)[:affected]})
        if st.button("提交调整",type="primary"):
            try:
                event=run.limit(kind,target,affected,clock(start),END+1 if keep else clock(end))
                save();st.success(f"已登记 {event}，在 {hhmm(clock(start))} 生效。")
            except ValueError as e:st.error(str(e))
    records=[e for e in run.log if e["type"] in ("limit","restore","demand")]
    if records:
        st.markdown("#### 已生效的调整")
        st.dataframe(records,hide_index=True)
    active=sorted({event for limits in run.cal.limits.values() for event in limits})
    if active:
        selected=st.selectbox("当前有效限制",active)
        if st.button("在下一分钟恢复"):
            run._push(run.now+1,1,"restore",selected)
            run.version+=1
            save();st.success("已排入下一分钟的事件队列")
    pending=[{"时刻":hhmm(row[0]),"事件":row[3],"编号":row[4]} for row in sorted(run.events) if row[3] in ("limit","restore")]
    if pending: st.dataframe(pending,hide_index=True)


def render_results(rows):
    if not rows:
        return
    frame=pd.DataFrame(rows)
    columns={"scenario":"场景","strategy":"策略","seed":"种子","arrivals":"到达",
             "completed":"交付","on_time":"准时","on_time_rate":"全体准时率","ground_on_time_rate":"仅地面准时率",
             "completion_rate":"完成率","p90_minutes":"P90 分钟","mean_wait":"平均待派分钟",
             "mean_launch_wait":"起飞前等待","ground":"地面","air":"空地",
             "switched":"转地面量","rider_minutes":"骑手人分钟"}
    chosen=frame[list(columns)].rename(columns=columns)
    chosen["场景"]=chosen["场景"].map(SCENARIOS)
    chosen["策略"]=chosen["策略"].map(STRATEGIES)
    st.dataframe(chosen,hide_index=True,column_config={key:st.column_config.NumberColumn(format="percent")
                 for key in ("全体准时率","仅地面准时率","完成率")})
    if len(chosen):
        from plotly import express as px
        melted=chosen.melt(id_vars=["策略","场景"],value_vars=["全体准时率","仅地面准时率"],var_name="指标",value_name="比例")
        chart=melted.groupby(["策略","场景","指标"],as_index=False)["比例"].mean()
        fig=px.bar(chart,x="策略",y="比例",color="指标",facet_col="场景",barmode="group",
                   color_discrete_sequence=["#0E766E","#5C7AA3"])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig,width="stretch")
    comparisons=[item for key in ("on_time_rate","ground_on_time_rate")
                 for style in ("ground","fixed","dynamic") for item in paired(rows,key,style)]
    if comparisons:
        st.markdown("#### 多订单试排的配对差")
        st.dataframe(comparisons,hide_index=True)
    st.download_button("下载批量结果 CSV",frame.drop(columns=["resources","rider_detail"]).to_csv(index=False).encode("utf-8-sig"),"batch.csv","text/csv")
    st.download_button("下载完整指标和输入哈希",encode(rows),"batch.json","application/json")


def replay_page():
    s=state()
    st.subheader("回放与对照")
    a,b=st.tabs(["逐单记录","策略对照"])
    with a:
        c=st.columns(2)
        ids=s.store.list()
        if ids:
            chosen=c[0].selectbox("运行编号",ids)
            if c[0].button("载入运行"):
                set_run(s.store.load(chosen));st.rerun()
        imported=c[1].file_uploader("恢复下载的运行包",type="zip")
        if imported and c[1].button("恢复运行"):
            try:
                set_run(s.store.import_bundle(imported.getvalue()))
                st.rerun()
            except (KeyError,ValueError,TypeError,OSError,zipfile.BadZipFile) as e:st.error(f"无法恢复运行：{e}")
        run=s.get("run")
        if run:
            metric_strip(run)
            decision_ids=[k for k,r in run.records.items() if r["status"] not in ("scheduled","removed")]
            if decision_ids:
                id=st.selectbox("订单编号",decision_ids)
                r=run.records[id]
                st.write(f"状态 **{STATUS[r['status']]}**　实际方式 **{MODES.get(r['mode'],'—')}**　送达 **{hhmm(r['actual'])}**")
                decisions=[d for d in run.decisions if d["order"]==id]
                if decisions:
                    index=st.select_slider("定位决策",options=list(range(len(decisions))),value=len(decisions)-1,
                                           format_func=lambda i:f"{hhmm(decisions[i]['minute'])} · {MODES[decisions[i]['mode']]}")
                    show_decision(decisions[index])
                elif "arrival_estimate" in r:
                    estimate=r["arrival_estimate"]
                    st.write(f"到达时估计 · 地面 {hhmm(estimate['ground_eta'])} / 空地 {hhmm(estimate['air_eta'])}")
                reservations=[asdict(slot) for group in run.cal.slots.values() for slot in group if slot.order==id]
                if reservations:
                    with st.expander("实际资源预约"):
                        st.dataframe(reservations,hide_index=True)
                records=[{"时刻":hhmm(e["minute"]),"事件":e["type"],"编号":e["id"]}
                         for e in run.log if e["id"]==id]
                st.dataframe(records,hide_index=True)
            st.download_button("下载运行包",s.store.export(run),f"run-{run.run_id}.zip","application/zip")
            exports=[{"订单":k,"到达":r["order"].arrival,"截止":r["order"].deadline,"送达":r["actual"],
                      "状态":STATUS[r["status"]],"配送":MODES.get(r["mode"],""),"仅地面":r["order"].ground_only}
                     for k,r in run.records.items() if r["status"] not in ("scheduled","removed")]
            if exports:st.download_button("导出逐单结果",as_csv(exports),f"orders-{run.run_id}.csv","text/csv")
            final=metrics(run)
            st.download_button("导出当前指标",encode(final),f"metrics-{run.run_id}.json","application/json")
    with b:
        st.write("四种策略共享订单、资源配置、航网和事件。")
        if st.button("比较同一起点的电池扰动"):
            pairs=[]
            for affected in (False,True):
                run=battery_pair(affected)
                run.advance(END)
                for id,r in run.records.items():
                    pairs.append({"场景":"停用 3 块" if affected else "原状态","订单":id,"实际方式":MODES.get(r["mode"],"待派"),
                                  "送达":hhmm(r["actual"]),"截止":hhmm(r["order"].deadline),
                                  "准时":r["actual"] is not None and r["actual"]<=r["order"].deadline})
            s.demo_pair=pairs
        if "demo_pair" in s:
            st.dataframe(s.demo_pair,hide_index=True)
        if "batch" not in s or s.batch["future"].done():
            with st.form("batch"):
                scenes=st.multiselect("实验场景",list(SCENARIOS),default=["normal"],format_func=SCENARIOS.get)
                days=st.number_input("模拟日数量",1,30,1)
                go_batch=st.form_submit_button("执行四策略对照",type="primary")
            if go_batch:
                if not scenes:
                    st.warning("请至少选择一个场景。")
                else:
                    net=network()
                    pr=profile()
                    cancel=Event()
                    rows=[]
                    chosen_hub=target_hub(net,s.get("planning",[]))
                    future=worker_pool().submit(run_batch,pr,net,range(201,201+days),scenes,
                                                hub=chosen_hub,cancelled=cancel.is_set,progress=rows.append)
                    s.batch={"future":future,"rows":rows,"cancel":cancel,"total":days*len(scenes)*4}
                    st.rerun()
        if "batch" in s:
            @st.fragment(run_every=None if s.batch["future"].done() else 1.)
            def status():
                job=s.batch
                st.progress(len(job["rows"])/job["total"],text=f"已完成 {len(job['rows'])} / {job['total']} 次")
                if not job["future"].done():
                    if st.button("取消批量任务"):job["cancel"].set()
                else:
                    if job["cancel"].is_set():st.info("已取消，保留已完成结果。")
                    if exc:=job["future"].exception():st.error(f"批量运行出错：{exc}")
                    render_results(job["rows"].copy())
            status()


STYLE = """
<style>
.block-container {max-width: 1480px; padding-top: 2.5rem;}
.app-title {font-size: clamp(1.4rem, 2.5vw, 2.35rem);font-weight:750;line-height:1.25;letter-spacing:-.035em;color:#122338;}
.app-subtitle {font-size: .95rem;color:#536780;margin-top:.45rem;margin-bottom:1.35rem;}
.recommend {background:#FFFFFF;border-left:5px solid;border-radius:8px;padding:.65rem 1rem;margin:1rem 0 .85rem;}
.recommend span {color:#536780;font-size:.9rem;display:block;}
.recommend strong {font-size:1.3rem;color:#122338;}
div[data-testid=stMetric] {background:#FFFFFF;}
</style>
"""


def main():
    st.set_page_config(page_title=TITLE,layout="wide",page_icon="✈️")
    st.markdown(STYLE,unsafe_allow_html=True)
    s=state()
    if "store" not in s:
        if CLOUD:
            s.temp=tempfile.TemporaryDirectory(prefix="dispatch-")
            s.store=RunStore(s.temp.name)
        else:
            s.store=RunStore(os.environ.get("DRONE_DISPATCH_RUNS",ROOT/"runs"))
        s.play=False
        s.origin_name="公开模拟 CSV"
        s.draft=None
    with st.sidebar:
        st.markdown("### 空地协同调度")
        st.caption("云端演示" if CLOUD else "本地模式")
        page=st.radio("工作台",PAGES,label_visibility="collapsed")
        if s.get("run"):
            st.divider()
            st.caption("当前运行")
            st.code(s.run.run_id,language=None)
            st.write(STRATEGIES[s.run.strategy])
        if CLOUD:
            st.caption("运行包可下载并恢复。云端会话结束后，请使用已下载的记录包。")
    st.markdown(f'<div class="app-title">{TITLE}</div><div class="app-subtitle">{ENGLISH}</div>',unsafe_allow_html=True)
    if page not in ["运行控制"] and s.play:
        s.play=False
    handler=dict(zip(PAGES,(decision_page,run_controls,network_page,data_page,disruption_page,replay_page)))
    try:
        handler[page]()
    except (OSError,ValueError) as e:
        st.error(str(e))
