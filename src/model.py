from dataclasses import dataclass, replace
from math import ceil, cos, hypot, radians
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
class Network:
    origin: tuple[float, float]
    bounds: tuple[float, float, float, float]
    routes: tuple[Route, ...]
    candidates: tuple[dict, ...] = ()
    diagnostics: dict | None = None

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
        return cls(tuple(row["origin"]), tuple(row["bounds"]), tuple(
            Route(**{**r, "pickup": tuple(r["pickup"]), "dropoff": tuple(r["dropoff"])})
            for r in row["routes"]
        ), tuple(row.get("candidates", [])), row.get("diagnostics"))


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
    if order.kg > 2:
        return replace(order, route_id=None, eligibility="包裹超过 2 kg")
    if not .5 <= network.straight(order.pickup, order.dropoff) <= 5:
        return replace(order, route_id=None, eligibility="取送直线距离不在 0.5 至 5 km")
    eligible = []
    for route in network.routes:
        first = network.road(order.pickup, route.pickup)
        last = network.road(route.dropoff, order.dropoff)
        if max(first, last) <= 2.5:
            nominal = minutes(first, 18)+minutes(last, 18)+minutes(route.km, 40)+6
            eligible.append((nominal, route.id))
    if eligible:
        return replace(order, route_id=min(eligible)[1], eligibility="可接入航网")
    return replace(order, route_id=None, eligibility="两端接驳范围内无可用航段")
