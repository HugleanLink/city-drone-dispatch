"""Empirical ground delivery times estimated from the selected planning source."""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, cos, floor, radians

import numpy as np
import pandas as pd

from .model import haversine
from .storage import encode
from hashlib import sha256

DISTANCES = (.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10)
PERIODS = (6, 10, 14, 17, 21)
CELL = 1.5
MIN_SAMPLES = 40
LEVELS = (
    ("period", "band", "business", "pr", "pc", "dr", "dc"),
    ("period", "band", "business", "pr", "pc"),
    ("hour", "band", "business"),
    ("period", "band", "business"),
    ("band", "business"),
    ("band",),
)


def features(profile):
    rows = profile.rows.loc[profile.rows.reference_minutes.gt(0)].copy()
    p_lat, p_lon, d_lat, d_lon = (rows[name].to_numpy() for name in (
        "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"))
    lat1, lon1, lat2, lon2 = np.radians([p_lat, p_lon, d_lat, d_lon])
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    km = 2*6371*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    lat, lon = profile.origin
    scale = 111*cos(radians(lat))
    hour = ((rows.minute + rows.pre_pickup) % 1440 // 60).astype(int)
    result = pd.DataFrame({"hour": hour, "period": np.searchsorted(PERIODS, hour, side="right"),
                           "payhour": rows.minute // 60, "band": np.searchsorted(DISTANCES, km),
                           "business": rows.biz_line, "pr": np.floor((p_lat-lat)*111/CELL).astype(int),
                           "pc": np.floor((p_lon-lon)*scale/CELL).astype(int),
                           "dr": np.floor((d_lat-lat)*111/CELL).astype(int),
                           "dc": np.floor((d_lon-lon)*scale/CELL).astype(int),
                           "duration": rows.reference_minutes, "pre": rows.pre_pickup})
    usable = result.loc[km >= .1]
    return usable.loc[usable.index.repeat(rows.loc[usable.index, "count"])]


def fit(profile):
    frame = features(profile)
    if frame.empty:
        raise ValueError("所选规划来源没有可用于地面估时的有效取送记录")
    tables = []
    for fields in LEVELS:
        grouped = frame.groupby(list(fields), observed=True).duration
        eligible = grouped.size()
        eligible = eligible.index[eligible >= MIN_SAMPLES]
        quantiles = grouped.quantile([.5, .9]).unstack().reindex(eligible)
        tables.append({key if isinstance(key, tuple) else (key,): tuple(map(float, row))
                       for key, row in zip(quantiles.index, quantiles.to_numpy())})
    grouped = frame.groupby(["business", "payhour"], observed=True).pre
    prep = grouped.quantile(.25)[grouped.size() >= MIN_SAMPLES].to_dict()
    default_prep = frame.groupby("business", observed=True).pre.quantile(.25).to_dict()
    reference = tuple(map(float, frame.duration.quantile([.5, .9])))
    checksum = sha256(encode([profile.origin, tuple((tuple(table.items()) for table in tables)),
                              tuple(prep.items()), default_prep, reference]).encode()).hexdigest()
    return GroundTimes(profile.origin, tuple(tables), prep, default_prep, reference, len(frame), checksum)


@dataclass(eq=False)
class GroundTimes:
    origin: tuple
    tables: tuple
    prep: dict
    default_prep: dict
    reference: tuple
    samples: int
    version: str

    def ready(self, order):
        return order.arrival + ceil(self.prep.get((order.business, order.arrival // 60),
                                                   self.default_prep.get(order.business, 8)))

    @lru_cache(maxsize=32768)
    def estimate(self, origin, destination, hour, business):
        lat, lon = self.origin
        scale = 111*cos(radians(lat))/CELL
        fields = dict(period=int(np.searchsorted(PERIODS, hour, side="right")), hour=hour, business=business,
                      band=int(np.searchsorted(DISTANCES, haversine(origin, destination))),
                      pr=floor((origin[0]-lat)*111/CELL), pc=floor((origin[1]-lon)*scale),
                      dr=floor((destination[0]-lat)*111/CELL), dc=floor((destination[1]-lon)*scale))
        for keys, table in zip(LEVELS, self.tables):
            key = tuple(fields[field] for field in keys)
            if key in table:
                return table[key]
        return self.reference

    def loaded(self, a, b, depart, business):
        if a == b:
            return 1
        return max(1, ceil(self.estimate(a, b, depart % 1440 // 60, business)[0]))

    def empty(self, a, b, depart, business):
        if a == b:
            return 0
        hour = depart % 1440 // 60
        local = self.estimate(a, (a[0] + .001, a[1]), hour, business)[0]
        return max(1, ceil(self.estimate(a, b, hour, business)[0] - local))
