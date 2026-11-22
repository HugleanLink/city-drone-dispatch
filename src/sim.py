import heapq
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import isfinite
from uuid import uuid4

from .model import (BUSINESS, END, LAST_ARRIVAL, START, STRATEGIES, Network, Order,
                    ground_time, match_order, minutes)


@dataclass(frozen=True)
class Slot:
    resource: str
    start: int
    end: int
    activity: str
    order: str


@dataclass(frozen=True)
class Plan:
    mode: str
    first: int
    eta: int
    fetch: int
    slots: tuple[Slot, ...]
    rider_ends: tuple
    route: str | None = None


class Calendar:
    def __init__(self):
        self.slots = defaultdict(list)
        self.limits = defaultdict(dict)
        self.tails = {}

    def fork(self):
        obj = Calendar()
        obj.slots.update((key, value.copy()) for key, value in self.slots.items())
        obj.limits.update(self.limits)
        obj.tails.update(self.tails)
        return obj

    def next(self, resources, start, length, extra=()):
        best = (END+1, "")
        for resource in resources:
            cursor = start
            intervals = [(s.start,s.end) for s in self.slots.get(resource,())]
            intervals.extend((s.start,s.end) for s in extra if s.resource==resource)
            intervals.extend(self.limits.get(resource,{}).values())
            for left,right in sorted(intervals):
                if right<=cursor:
                    continue
                if left>=cursor+length:
                    break
                cursor = right
            if cursor < END and cursor+length <= END:
                best = min(best,(cursor, resource))
        return best

    def free(self, resource, start, end, extra=()):
        return self.next([resource], start, end-start, extra)[0] == start

    def reserve(self, plan):
        for slot in plan.slots:
            self.slots[slot.resource].append(slot)
        for rider, end, loc in plan.rider_ends:
            self.tails[rider] = (end, loc)


class Engine:
    def __init__(self, network, orders=(), strategy="lookahead", settings=None, start=START):
        if strategy not in STRATEGIES:
            raise ValueError("未知调度策略")
        self.network, self.strategy = network, strategy
        self.settings = {"riders":4,"aircraft":4,"battery":6,"pads":1,"swap":1,"charge":2,**(settings or {})}
        if any(isinstance(n,bool) or not isinstance(n,int) or not 0<=n<=100 for n in self.settings.values()):
            raise ValueError("资源数量应为 0 至 100 的整数")
        self.run_id = uuid4().hex[:12]
        self.now, self.version, self.seq = start-1, 0, 0
        self.events = []
        self.records = {}
        self.requests = {}
        self.decisions = []
        self.log = []
        self.cal = Calendar()
        self.hubs, self.pads = {}, {}
        self.routes = {r.id:r for r in network.routes}
        self._resources(start)
        for order in orders:
            self.add(order)

    def _resources(self, start):
        def ids(kind, site, count):
            return [f"{kind}:{site}:{i+1:02}" for i in range(count)]
        left,right,bottom,top = self.network.bounds
        for zone,(x,y) in enumerate(((left/2,bottom/2),(left/2,top/2),(right/2,bottom/2),(right/2,top/2)),1):
            for rider in ids("rider",str(zone),self.settings["riders"]):
                self.cal.tails[rider] = (start,self.network.latlon(round(x*2)/2,round(y*2)/2))
        for route in self.routes.values():
            hub_site = route.hub_site or route.hub
            spoke_site = route.spoke_site or route.spoke
            for site in [hub_site,spoke_site]:
                self.pads[site] = ids("pad",site,self.settings["pads"])
            if route.hub not in self.hubs:
                planes=ids("aircraft",route.hub,self.settings["aircraft"])
                self.hubs[route.hub] = {
                    "aircraft": planes, "mounted": {plane:f"mounted:{plane}" for plane in planes},
                    "battery": ids("battery",route.hub,self.settings["battery"]),
                    "swap": ids("swap",route.hub,self.settings["swap"]),
                    "charge": ids("charge",route.hub,self.settings["charge"]),
                }

    def _push(self, when, priority, kind, object_id, payload=None):
        self.seq += 1
        heapq.heappush(self.events,(when,priority,self.seq,kind,object_id,payload))

    def validate(self, order):
        if not max(START,self.now+1)<=order.arrival<LAST_ARRIVAL:
            raise ValueError("下单时间须位于接单窗口内且尚未处理")
        if not order.arrival<=order.ready<=order.deadline or not isfinite(order.kg) or order.kg<=0 or order.business not in BUSINESS:
            raise ValueError("请检查业务、重量、备妥时间和交付时限")
        return match_order(self.network,order)

    def add(self, order, request=None):
        if request and request in self.requests:
            return self.requests[request]
        order = self.validate(order)
        if order.id in self.records:
            raise ValueError("订单编号已存在")
        self.records[order.id] = {"order":order, "status":"scheduled", "intent":None,
                                  "first_intent":None,"mode":None,"actual":None}
        self._push(order.arrival,2,"arrival",order.id)
        if order.ready != order.arrival:
            self._push(order.ready,2,"ready",order.id)
        if request:
            self.requests[request]=order.id
        self.version+=1
        return order.id

    def _rider_options(self, cal, origin, destination, ready, now, force, extra=(), tails=None):
        tails = tails if tails is not None else cal.tails
        for rider,(available,position) in sorted(tails.items()):
            if force and max(ready,available)>now:
                continue
            start=max(now,ready,available)
            empty=ground_time(self.network.road(position,origin),start)
            collected=start+empty+1
            end=collected+ground_time(self.network.road(origin,destination),collected)+1
            if end>END:
                continue
            # Each rider's assignments append to their committed tail.
            yield rider,start,collected,end,empty

    def _ground(self, order, cal, now, force):
        options=self._rider_options(cal,order.pickup,order.dropoff,order.ready,now,force)
        best=min(options,key=lambda r:(r[3],r[1],r[0]),default=None)
        if best is None:
            return None
        rider,start,fetch,end,empty=best
        slots=self._rider_slots(order.id,rider,start,end,empty,"地面直送")
        return Plan("ground",start,end,fetch,slots,((rider,end,order.dropoff),))

    @staticmethod
    def _rider_slots(order_id,rider,start,end,empty,activity):
        travel=(Slot(rider,start,start+empty,"空驶",order_id),) if empty else ()
        return travel+(Slot(rider,start+empty,end,activity,order_id),)

    def _air(self, order, cal, now, force):
        route=self.routes.get(order.route_id)
        if not route:
            return None
        resources=self.hubs[route.hub]
        hub_pads=self.pads[route.hub_site or route.hub]
        spoke_pads=self.pads[route.spoke_site or route.spoke]
        if not (hub_pads and spoke_pads and resources["aircraft"] and resources["swap"]):
            return None
        fly=max(1,minutes(route.km,40))
        first_legs=sorted(self._rider_options(cal,order.pickup,route.pickup,order.ready,now,force),
                          key=lambda x:(x[3],x[0]))
        best=None
        # End-to-end minimum over feasible first rider, plane and final rider.
        # When the earliest feasible chain is already worse, later launches cannot improve it.
        for first,start,fetch,at_hub,empty in first_legs:
            if best and at_hub+fly+5>best.eta:
                break
            first_slots=self._rider_slots(order.id,first,start,at_hub,empty,"首段接驳")
            tails={**cal.tails, first:(at_hub,route.pickup)}
            for plane in resources["aircraft"]:
                plane_slots=cal.slots.get(plane,())
                plane_ready=max([now]+[slot.end for slot in plane_slots])
                mounted=resources["mounted"][plane]
                batteries=[mounted] if not plane_slots and not cal.slots.get(mounted) else resources["battery"]+[b for b in resources["mounted"].values() if cal.slots.get(b)]
                for battery in batteries:
                    already_mounted=not plane_slots and battery==mounted and not cal.slots.get(mounted)
                    batt_ready=max([now]+[s.end for s in cal.slots.get(battery,())])
                    takeoff=max(at_hub,plane_ready,batt_ready)
                    mount=()
                    if not already_mounted:
                        mount_time,swap_id=cal.next(resources["swap"],max(plane_ready,batt_ready),2)
                        if not swap_id:
                            continue
                        mount=(Slot(swap_id,mount_time,mount_time+2,"装电",order.id),)
                        takeoff=max(takeoff,mount_time+2)
                    latest=min(END-2*fly-6,best.eta-fly-4 if best else END)
                    while takeoff<=latest:
                        launch,pad=cal.next(hub_pads,takeoff,1,mount)
                        if not pad or launch>latest:
                            break
                        touch=launch+1+fly
                        receive,spoke_pad=cal.next(spoke_pads,touch,3,mount)
                        if not spoke_pad:
                            break
                        if receive>touch:
                            takeoff=launch+receive-touch
                            continue
                        deliver_ready=touch+2
                        prelim=mount+(Slot(pad,launch,launch+1,"载货起飞",order.id),)
                        return_land,landing=cal.next(hub_pads,deliver_ready+1+fly,1,prelim)
                        if not landing:
                            break
                        return_start=return_land-fly-1
                        spoke_slot=Slot(spoke_pad,touch,return_start+1,"降落、卸货及返程起飞",order.id)
                        if not cal.free(spoke_pad,touch,spoke_slot.end,prelim):
                            receive, _ = cal.next([spoke_pad],touch+1,spoke_slot.end-touch,prelim)
                            takeoff=launch+max(1,receive-touch)
                            continue
                        unload,swap_id=cal.next(resources["swap"],return_land+1,2,mount)
                        if not swap_id:
                            break
                        aircraft_start=mount[0].start if mount else launch
                        if not cal.free(plane,aircraft_start,unload+2):
                            if mount:
                                # Mounting before a blackout would keep the plane reserved.
                                new_mount,swap=cal.next([mount[0].resource],max(launch+1,plane_ready,batt_ready),2)
                                if not swap:
                                    break
                                mount=(Slot(swap,new_mount,new_mount+2,"装电",order.id),)
                                takeoff=max(launch+1,new_mount+2)
                            else:
                                takeoff=launch+1
                            continue
                        if not cal.free(battery,aircraft_start,unload+2):
                            break
                        charge,charger=cal.next(resources["charge"],unload+2,24)
                        charged=charge+24 if charger else END+1
                        # A future restriction blocks reuse, not protected work already in flight.
                        battery_slot=Slot(battery,aircraft_start,charged,"绑定与周转",order.id)
                        extra=prelim+(spoke_slot,Slot(landing,return_land,return_land+1,"返场降落",order.id),
                             Slot(swap_id,unload,unload+2,"卸电",order.id),
                             Slot(plane,aircraft_start,unload+2,"航空任务",order.id),battery_slot)
                        if charger:
                            extra+=(Slot(charger,charge,charged,"充电",order.id),)
                        for last,(available,loc) in sorted(tails.items()):
                            begin=max(now,available)
                            distance=self.network.road(loc,route.dropoff)
                            travel=ground_time(distance,begin)
                            if begin+travel < deliver_ready:
                                begin=max(begin,deliver_ready-travel)
                                travel=ground_time(distance,begin)
                            transfer=max(deliver_ready,begin+travel)
                            eta=transfer+1+ground_time(self.network.road(route.dropoff,order.dropoff),transfer+1)+1
                            if eta>END or (best and (eta,start,first,plane,last)>=(best.eta,best.first,best.rider_ends[0][0],
                                    next(s.resource for s in best.slots if s.activity=="航空任务"),best.rider_ends[-1][0])):
                                continue
                            last_slots=self._rider_slots(order.id,last,begin,eta,travel,"末段接驳")
                            best=Plan("air",start,eta,fetch,first_slots+extra+last_slots,
                                      ((first,at_hub,route.pickup),(last,eta,order.dropoff)),route.id)
                        break
        return best

    def opts(self, order, cal=None, now=None, force=False):
        cal=self.cal if cal is None else cal
        now=self.now if now is None else now
        return self._ground(order,cal,now,force),self._air(order,cal,now,force)

    @staticmethod
    def _rank(order, options):
        punctual=[plan for plan in options if plan and plan.eta<=order.deadline]
        if len(punctual)==1:
            return (0,order.deadline-punctual[0].eta,order.deadline,order.ready,order.id)
        return (1 if punctual else 2,0,order.deadline,order.ready,order.id)

    @staticmethod
    def _best(options):
        return min((p for p in options if p),key=lambda p:(p.eta,p.mode!="ground",p.first),default=None)

    def evaluate(self, order, others=(), now=None):
        now=self.now if now is None else now
        normal=self.opts(order,now=now)
        first_ready=max(now,order.ready)
        immediate=self.opts(order,now=first_ready,force=True)
        branches=[]
        for mode, fixed in [("ground",immediate[0]),("air",immediate[1]),("wait",None)]:
            if mode!="wait" and fixed is None:
                continue
            calendar=self.cal.fork()
            queue=list(others)
            projected={}
            if fixed:
                calendar.reserve(fixed)
                projected[order.id]=(order,fixed)
            else:
                queue.append(order)
            while queue:
                choices=[(row,self.opts(row,calendar,max(now,first_ready+1) if row.id==order.id else now))
                         for row in queue if row.deadline>=now]
                if not choices:
                    # Late work is last in the priority sequence, so it cannot improve a punctual score.
                    # Keep a possible immediate alternative to dispatch when the waiting branch wins.
                    choice_rows=[row for row in queue if row.id==order.id or (mode=="wait" and not projected)]
                    choices=[(row,self.opts(row,calendar,max(now,first_ready+1) if row.id==order.id else now))
                             for row in choice_rows]
                    if not choices:
                        break
                row,options=min(choices,key=lambda pair:self._rank(*pair))
                queue.remove(row)
                selected=self._best(options)
                if selected:
                    calendar.reserve(selected)
                    projected[row.id]=(row,selected)
            timely=[row for row,p in projected.values() if p.eta<=row.deadline]
            own=projected.get(order.id,(None,None))[1]
            alternative=None
            if mode=="wait":
                startable=[(row,p) for row,p in projected.values() if row.id!=order.id and row.ready<=now and p.first==now]
                if startable:
                    alternative=min(startable,key=lambda pair:self._rank(pair[0],self.opts(pair[0],now=now)))
            branches.append({"mode":mode,"on_time":len(timely),
                             "ground_on_time":sum(o.ground_only for o in timely),
                             "on_time_ids":sorted(o.id for o in timely),
                             "eta":own.eta if own else None,"plan":fixed,"alternative":alternative})
        top=min(branches,key=lambda b:(-b["on_time"],-b["ground_on_time"],b["mode"]=="wait",
                                      b["eta"] if b["eta"] is not None else END+1,
                                      b["mode"]!="ground"))
        winning=set(top["on_time_ids"])
        summary=[{k:v for k,v in b.items() if k not in ("plan","alternative")} |
                 {"loses_vs_recommendation":sorted(winning-set(b["on_time_ids"])),
                  "gains_vs_recommendation":sorted(set(b["on_time_ids"])-winning)}
                 for b in branches]
        guide={"order":order.id,"minute":now,"version":self.version,
               "ground_eta":normal[0].eta if normal[0] else None,"air_eta":normal[1].eta if normal[1] else None,
               "deadline":order.deadline,"mode":top["mode"],"branches":summary,
               "air_reason":self._air_reason(order,normal[1]),"ground_reason":"可排" if normal[0] else "骑手不足或超出仿真时窗"}
        return guide,top

    def _air_reason(self,order,plan):
        if plan:
            return "预计准时" if plan.eta<=order.deadline else "预计超过截止"
        if order.ground_only:
            return order.eligibility
        route=self.routes[order.route_id]
        resources=self.hubs[route.hub]
        if not resources["aircraft"]:
            return "无可用飞机"
        if not self.pads[route.hub_site or route.hub] or not self.pads[route.spoke_site or route.spoke]:
            return "无可用起降位"
        if not resources["swap"]:
            return "无可用换电位"
        if not self.cal.tails:
            return "无可用骑手"
        return "飞机、电池、起降位或接驳资源无法在时窗内完成预约"

    def preview(self,order):
        order=self.validate(order)
        now=max(order.arrival,self.now)
        others=[record["order"] for record in self.records.values() if record["status"] in ("waiting","unready")]
        result,_=self.evaluate(order,others,now=now)
        return result

    def _intent(self, order, guide):
        record=self.records[order.id]
        mode=guide["mode"]
        record["intent"]=mode
        if record["first_intent"] is None:
            record["first_intent"]=mode
        if mode=="air":
            record["ever_air"]=True
        previous=record.get("decision")
        keys=("mode","ground_eta","air_eta","air_reason")
        if not previous or any(guide.get(k)!=previous.get(k) for k in keys):
            self.decisions.append(guide)
            record["decision"]=guide

    def _commit(self,order,plan):
        self.cal.reserve(plan)
        self.records[order.id].update(status="assigned",mode=plan.mode,first=plan.first,fetch=plan.fetch,eta=plan.eta)
        self._push(plan.eta,0,"delivered",order.id)
        self.version+=1
        self.log.append({"minute":self.now,"type":"dispatch","id":order.id,"mode":plan.mode})

    def dispatch(self):
        while True:
            if not any(end<=self.now for end,_ in self.cal.tails.values()):
                return
            ready=[r["order"] for r in self.records.values() if r["status"]=="waiting"]
            if not ready:
                return
            if self.strategy in ("ground","fixed"):
                options=((o,(self._ground(o,self.cal,self.now,False),None) if self.strategy=="ground" else self.opts(o))
                         for o in sorted(ready,key=lambda o:(o.ready,o.id)))
            else:
                options=[(o,self.opts(o)) for o in ready]
                options.sort(key=lambda pair:self._rank(*pair))
            dispatched=False
            for o, plans in options:
                r=self.records[o.id]
                if self.strategy=="ground":
                    choice=plans[0]
                elif self.strategy=="fixed":
                    if "fixed_mode" not in r:
                        target=self.routes[o.route_id].hub if o.route_id else None
                        queued=sum(1 for record in self.records.values()
                                   if record.get("fixed_hub")==target and record.get("fixed_mode")=="air"
                                   and record["status"]=="waiting")
                        r.update(fixed_hub=target,fixed_mode="air" if target and queued<8 else "ground")
                    choice=plans[1] if r["fixed_mode"]=="air" else plans[0]
                else:
                    choice=self._best(plans)
                can_start=self.opts(o,force=True) if self.strategy=="lookahead" and choice and choice.first!=self.now else None
                if self.strategy=="lookahead" and (choice and choice.first==self.now or can_start and any(can_start)):
                    others=[item["order"] for key,item in self.records.items() if key!=o.id and item["status"] in ("waiting","unready")]
                    guide,branch=self.evaluate(o,others)
                    self._intent(o,guide)
                    if branch["plan"]:
                        self._commit(o,branch["plan"]);dispatched=True
                    elif branch["alternative"]:
                        alternate,plan=branch["alternative"]
                        actual=self.opts(alternate,force=True)
                        self._intent(alternate,{"order":alternate.id,"minute":self.now,
                                               "version":self.version,"mode":plan.mode,
                                               "ground_eta":actual[0].eta if actual[0] else None,
                                               "air_eta":actual[1].eta if actual[1] else None,
                                               "deadline":alternate.deadline,
                                               "air_reason":self._air_reason(alternate,actual[1]),"branches":[]})
                        self._commit(alternate,plan)
                        dispatched=True
                    break
                self._intent(o,{"order":o.id,"minute":self.now,"version":self.version,"mode":choice.mode if choice else "wait",
                                "ground_eta":plans[0].eta if plans[0] else None,"air_eta":plans[1].eta if plans[1] else None,
                                "deadline":o.deadline,"air_reason":self._air_reason(o,plans[1]),"branches":[]})
                if choice and choice.first==self.now:
                    self._commit(o,choice);dispatched=True
                    break
            if not dispatched:
                return

    def resources(self,kind,site):
        if kind=="pad":
            return self.pads.get(site,[])
        if kind not in ("aircraft","battery"):
            raise ValueError("未知资源种类")
        return self.hubs.get(site,{}).get(kind,[])

    def limit(self, kind, site, count, start, end=END+1, fixed=None):
        if not self.now<start<LAST_ARRIVAL or not start<end<=END+1 or count<0:
            raise ValueError("请检查限制时间与数量")
        candidates=self.resources(kind,site)
        if fixed is not None and not set(fixed)<=set(candidates):
            raise ValueError("限制资源与站点不匹配")
        event_id=f"L{self.seq+1:05}"
        self._push(start,1,"limit",event_id,dict(kind=kind,site=site,count=count,end=end,fixed=fixed))
        if end<=END:
            self._push(end,1,"restore",event_id)
        self.version+=1
        return event_id

    def adjust_orders(self,wind_start,wind_end,target,additions):
        if not self.now<wind_start<wind_end<=LAST_ARRIVAL or target<0:
            raise ValueError("未来时窗或订单数量无效")
        pending=sorted((r["order"].id for r in self.records.values()
                        if r["status"]=="scheduled" and wind_start<=r["order"].arrival<wind_end), reverse=True)
        needed=max(0,target-len(pending))
        if len(additions)!=needed or any(not wind_start<=o.arrival<wind_end for o in additions):
            raise ValueError("追加订单与目标数量或时窗不符")
        if len({o.id for o in additions})!=needed or any(o.id in self.records for o in additions):
            raise ValueError("追加订单编号冲突")
        for o in additions:
            self.validate(o)
        removed=pending[:max(0,len(pending)-target)]
        for id in removed:
            self.records[id]["status"]="removed"
        for o in additions:
            self.add(o)
        self.version+=1
        self.log.append({"minute":self.now,"type":"demand","added":[o.id for o in additions],"removed":removed})

    def _event(self,kind,id,payload):
        if kind=="limit":
            candidates=self.resources(payload["kind"],payload["site"])
            active=lambda resource: any(s.end>self.now for s in self.cal.slots.get(resource,()))
            # Released, unbound spare batteries only for interactive impairments.
            selected=payload["fixed"]
            if selected is None:
                is_battery=payload["kind"]=="battery"
                limited=lambda resource: bool(self.cal.limits.get(resource))
                selected=sorted((x for x in candidates if not limited(x) and (not is_battery or not active(x))),
                                key=lambda x:(active(x),x))[:payload["count"]]
            for resource in selected:
                self.cal.limits[resource][id]=(self.now,payload["end"])
            self.log.append({"minute":self.now,"type":kind,"id":id,"resources":selected})
            return
        if kind=="restore":
            for active in self.cal.limits.values():
                active.pop(id,None)
            self.log.append({"minute":self.now,"type":kind,"id":id})
            return
        record=self.records[id]
        order=record["order"]
        if record["status"]=="removed":
            return
        if kind=="arrival":
            record["status"]="waiting" if order.ready<=self.now else "unready"
            if record["status"]=="unready":
                record["arrival_estimate"]=dict(zip(("ground_eta","air_eta"),(p.eta if p else None for p in self.opts(order))))
        elif kind=="ready" and record["status"]=="unready":
            record["status"]="waiting"
        elif kind=="delivered":
            record.update(status="delivered",actual=self.now)
        self.log.append({"minute":self.now,"type":kind,"id":id})

    def next_event(self):
        if self.now>=END:
            return END
        return min(self.events[0][0] if self.events else END,
                   self.now+1 if any(r["status"]=="waiting" for r in self.records.values()) else END)

    def advance(self, target, cancelled=lambda:False):
        if target<self.now or target>END:
            raise ValueError("目标分钟超出可推进范围")
        while self.now<target and not cancelled():
            step=min(target,self.next_event())
            self.now=step
            while self.events and self.events[0][0]<=self.now:
                _,_,_,kind,id,payload=heapq.heappop(self.events)
                self._event(kind,id,payload)
            if self.now<END:
                self.dispatch()
            else:
                for record in self.records.values():
                    if record["status"] in ("waiting","unready","assigned"):
                        record["status"]="incomplete"
            self.version+=1
        return self.now

    def snapshot(self):
        return {"schema_version":1,"run_id":self.run_id,"network":asdict(self.network),
                "strategy":self.strategy,"settings":self.settings,"now":self.now,"version":self.version,
                "seq":self.seq,"events":self.events,"requests":self.requests,
                "records":{k:{**r,"order":asdict(r["order"])} for k,r in self.records.items()},
                "slots":{k:[asdict(s) for s in slots] for k,slots in self.cal.slots.items()},
                "limits":dict(self.cal.limits),"tails":self.cal.tails,"decisions":self.decisions,"log":self.log}

    @classmethod
    def restore(cls,obj, fork=False):
        if obj["schema_version"]!=1:
            raise ValueError("运行包版本不支持")
        run=cls(Network.from_dict(obj["network"]),strategy=obj["strategy"],settings=obj["settings"])
        run.run_id=uuid4().hex[:12] if fork else obj["run_id"]
        run.now,run.version,run.seq=obj["now"],obj["version"],obj["seq"]
        run.events=[tuple(row) for row in obj["events"]]
        heapq.heapify(run.events)
        run.requests=obj["requests"]
        run.records={k:{**r,"order":Order.from_dict(r["order"])} for k,r in obj["records"].items()}
        run.cal.slots.update((k,[Slot(**s) for s in values]) for k,values in obj["slots"].items())
        run.cal.limits.update(obj["limits"])
        run.cal.tails={k:(v[0],tuple(v[1])) for k,v in obj["tails"].items()}
        run.decisions,run.log=obj["decisions"],obj["log"]
        return run
