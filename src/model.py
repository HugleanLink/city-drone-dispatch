from dataclasses import dataclass, replace
from math import asin, ceil, cos, hypot, isfinite, radians, sin, sqrt
from functools import lru_cache

BUSINESS = {
    "waimai": ("外卖", 4, 12, 35, .3, 1.5),
    "pinhaofan": ("拼好饭", 3, 10, 40, .3, 1.2),
    "shangou": ("闪购", 1, 6, 45, .5, 3.0),
    "health": ("医药健康", 1, 8, 45, .2, 1.5),
}
STRATEGIES = {
    "ground": "纯地面",
    "fixed": "固定分流",
    "dynamic": "逐单动态",
    "lookahead": "多订单试排",
}
MODES = {"ground": "地面直送", "air": "空地接驳", "wait": "等待复核"}
START, LAST_ARRIVAL, END = 600, 1200, 1320


@dataclass(frozen=True)
class Order:
    id: str
    arrival: int
    ready: int
    deadline: int
    pickup: tuple[float, float]
    dropoff: tuple[float, float]
    business: str
    kg: float
    price: float | None = None
    route_id: str | None = None
    eligibility: str = "未匹配"
    source: str = "generated"
    reference_minutes: float | None = None

    @property
    def ground_only(self):
        return self.route_id is None

    @classmethod
    def from_dict(cls, row):
        return cls(**{**row, "pickup": tuple(row["pickup"]), "dropoff": tuple(row["dropoff"])})


@dataclass(frozen=True)
class Route:
    id: str
    hub: str
    spoke: str
    pickup: tuple[float, float]
    dropoff: tuple[float, float]
    km: float
    coverage: int
    hub_site: str = ""
    spoke_site: str = ""


@dataclass(frozen=True)
class FlightRules:
    weight: float = 2.
    order_min: float = .5
    order_max: float = 5.
    access_km: float = 2.5
    leg_min: float = .5
    leg_max: float = 4.
    report: bool = False
    exclude_group_meal: bool = False

    def __post_init__(self):
        values = (self.weight, self.order_min, self.order_max, self.access_km, self.leg_min, self.leg_max)
        if not all(isfinite(value) for value in values) or not (
            self.weight > 0 and 0 <= self.order_min < self.order_max and self.access_km > 0
            and 0 < self.leg_min < self.leg_max
        ):
            raise ValueError("请检查重量、距离区间和接驳范围")


@lru_cache(maxsize=100_000)
def haversine(a, b):
    lat1, lon1, lat2, lon2 = map(radians, (*a, *b))
    h = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2 * 6371 * asin(sqrt(min(1., h)))


@dataclass(frozen=True)
class Network:
    origin: tuple[float, float]
    bounds: tuple[float, float, float, float]
    routes: tuple[Route, ...]
    candidates: tuple[dict, ...] = ()
    diagnostics: dict | None = None
    rider_starts: tuple = ()
    rules: FlightRules = FlightRules()

    def distance(self, a, b):
        if not self.rules.report:
            return self.straight(a, b)
        return haversine(a, b)

    def access(self, a, b):
        return self.distance(a, b) if self.rules.report else self.road(a, b)

    def rejection(self, order):
        rules = self.rules
        if order.kg > rules.weight:
            return f"包裹超过 {rules.weight:g} kg"
        if not rules.order_min <= self.distance(order.pickup, order.dropoff) <= rules.order_max:
            return f"取送直线距离不在 {rules.order_min:g} 至 {rules.order_max:g} km"
        if rules.exclude_group_meal and order.business == "pinhaofan":
            return "当前选址规则将拼好饭分配给地面配送"
        return None

    def xy(self, point):
        scale = 111 * cos(radians(self.origin[0]))
        return ((point[1]-self.origin[1])*scale, (point[0]-self.origin[0])*111)

    def latlon(self, x, y):
        return (self.origin[0]+y/111, self.origin[1]+x/(111*cos(radians(self.origin[0]))))

    def contains(self, point):
        x, y = self.xy(point)
        left, right, bottom, top = self.bounds
        return left <= x <= right and bottom <= y <= top

    def straight(self, a, b):
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        return hypot(ax-bx, ay-by)

    def road(self, a, b):
        a,b=tuple(a),tuple(b)
        return _road(self.origin, *sorted((a,b)))

    @classmethod
    def from_dict(cls, row):
        import re

        origin=tuple(row["origin"])
        bounds=tuple(row["bounds"])
        if (len(origin)!=2 or not all(isfinite(x) for x in origin)
                or abs(origin[0])>=85 or abs(origin[1])>180):
            raise ValueError("航网坐标参考无效")
        if len(bounds)!=4 or not all(isfinite(x) and abs(x)<=100 for x in bounds):
            raise ValueError("航网范围超出模拟城区限制")
        if bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:
            raise ValueError("航网范围顺序无效")
        rules = FlightRules(**row.get("rules", {}))
        labels=set()
        routes=[]
        for data in row["routes"]:
            route=Route(**{**data,"pickup":tuple(data["pickup"]),"dropoff":tuple(data["dropoff"])})
            for value in (route.id,route.hub,route.spoke):
                if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}",value):
                    raise ValueError("站点或航段编号格式无效")
            if (route.id in labels or not isfinite(route.km) or not rules.leg_min <= route.km <= rules.leg_max
                    or not all(len(point)==2 and all(isfinite(x) for x in point)
                               for point in (route.pickup,route.dropoff))):
                raise ValueError("航段重复或几何数据无效")
            routes.append(route)
            labels.add(route.id)
        starts=tuple(tuple(point) for point in row.get("rider_starts",[]))
        if starts and (len(starts)!=4 or not all(len(point)==2 and all(isfinite(x) for x in point) for point in starts)):
            raise ValueError("骑手起始坐标无效")
        candidates = tuple({**point, "location": tuple(point["location"])} if "location" in point else point
                           for point in row.get("candidates", ()))
        result = cls(origin, bounds, tuple(routes), candidates, row.get("diagnostics"), starts, rules)
        if any(not result.contains(point) for route in routes for point in (route.pickup,route.dropoff)):
            raise ValueError("航段端点位于航网范围之外")
        return result


@lru_cache(maxsize=100_000)
def _road(origin, a, b):
    if a==b:
        return 0.
    scale=111*cos(radians(origin[0]))
    def vector(point):
        return ((point[1]-origin[1])*scale,(point[0]-origin[0])*111)
    ax,ay=vector(a)
    bx,by=vector(b)
    an=[ceil(value*2-.5)/2 for value in (ax,ay)]
    bn=[ceil(value*2-.5)/2 for value in (bx,by)]
    return hypot(ax-an[0],ay-an[1])+abs(an[0]-bn[0])+abs(an[1]-bn[1])+hypot(bx-bn[0],by-bn[1])


def minutes(km, speed):
    return ceil(60*km/speed - 1e-10)


def ground_time(km, depart):
    return minutes(km, 13 if 690 <= depart < 810 or 1020 <= depart < 1200 else 18)


def match_order(network, order):
    if not network.contains(order.pickup) or not network.contains(order.dropoff):
        raise ValueError("取送点须在已载入路网的范围内")
    if reason := network.rejection(order):
        return replace(order, route_id=None, eligibility=reason)
    eligible = []
    for route in network.routes:
        if (network.access(order.pickup, route.pickup) <= network.rules.access_km
                and network.access(route.dropoff, order.dropoff) <= network.rules.access_km):
            first = network.road(order.pickup, route.pickup)
            last = network.road(route.dropoff, order.dropoff)
            nominal = minutes(first, 18)+minutes(last, 18)+minutes(route.km, 40)+6
            eligible.append((nominal, route.id))
    if eligible:
        return replace(order, route_id=min(eligible)[1], eligibility="可接入航网")
    return replace(order, route_id=None, eligibility="两端接驳范围内无可用航段")
