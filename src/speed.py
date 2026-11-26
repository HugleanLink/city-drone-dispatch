"""Distance screening and reproducible hourly effective delivery speeds."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .model import haversine


def mixture(values, minimum_components=1):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    scores = {}
    best = None
    maximum = min(6, len(values), len(np.unique(values)))
    with threadpool_limits(limits=1):
        for count in range(min(minimum_components, maximum), maximum + 1):
            if count < 1 or len(values) < 2:
                continue
            model = GaussianMixture(n_components=count, random_state=42).fit(values)
            score = model.bic(values)
            scores[count] = float(score)
            if best is None or score < scores[best.n_components]:
                best = model
    return best, scores


def distance_screen(distances):
    model, scores = mixture(distances, minimum_components=2)
    if model is not None:
        targets = np.flatnonzero((model.means_[:, 0] >= 1) & (model.means_[:, 0] <= 6))
        if len(targets):
            chosen = np.asarray(distances)[model.predict_proba(np.asarray(distances).reshape(-1, 1))[:, targets].sum(1) > .5]
            if len(chosen):
                low, high = (round(float(q), 2) for q in np.quantile(chosen, [.05, .95]))
                if low < high:
                    return low, high, len(chosen) / len(distances), scores, False
    return 1., 5., .5, scores, True


def distances_between(a, b):
    """Broadcastable great-circle distances for planning arrays."""
    a, b = np.radians(a), np.radians(b)
    lat = b[..., 0] - a[..., 0]
    lon = b[..., 1] - a[..., 1]
    h = np.sin(lat / 2) ** 2 + np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(lon / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


@dataclass(frozen=True)
class HourlySpeeds:
    # Minutes per kilometre, fitted once so decoders evaluate identical actions
    # against identical travel times on every pass.
    quantiles: tuple
    diagnostics: tuple

    def travel(self, start, end, minute, percentile=50):
        factor = self.quantiles[minute % 1440 // 60][percentile == 90]
        return ceil(haversine(start, end) * factor - 1e-10)


def fit_hourly(profile, minimum_samples=40):
    frame = profile.rows
    pickup = frame[["pickup_latitude", "pickup_longitude"]].to_numpy()
    dropoff = frame[["dropoff_latitude", "dropoff_longitude"]].to_numpy()
    km = distances_between(pickup, dropoff)
    duration = frame.reference_minutes.to_numpy(float)
    valid = (km >= .1) & (km <= 20) & (duration > 0) & (duration <= 180) & frame.pre_pickup.notna().to_numpy()
    if not valid.any():
        raise ValueError("没有可用于拟合有效配送速度的取货至送达记录")
    hours = ((frame.minute.to_numpy()[valid] + frame.pre_pickup.to_numpy()[valid]) % 1440 // 60).astype(int)
    counts = frame["count"].to_numpy(int)[valid]
    logs = np.log(60 * km[valid] / duration[valid])
    by_hour = tuple(np.repeat(logs[hours == hour], counts[hours == hour]) for hour in range(24))
    cached = {}
    quantiles, diagnostics = [], []
    for hour in range(24):
        source = (hour,)
        if len(by_hour[hour]) < minimum_samples:
            source = tuple(sorted({(hour - 1) % 24, hour, (hour + 1) % 24}))
            if sum(len(by_hour[h]) for h in source) < minimum_samples:
                source = tuple(range(24))
        if source not in cached:
            sample = np.concatenate([by_hour[h] for h in source])
            model, scores = mixture(sample)
            if model is None:
                draws = sample
            else:
                rng = np.random.default_rng(42)
                chosen = rng.choice(model.n_components, 20_000, p=model.weights_)
                draws = rng.normal(model.means_[chosen, 0], np.sqrt(model.covariances_[chosen, 0, 0]))
            cached[source] = (tuple(float(x) for x in np.quantile(60 * np.exp(-draws), [.5, .9])),
                              {"hours": list(source), "samples": len(sample), "bic": scores,
                               "components": model.n_components if model is not None else 1})
        estimates, details = cached[source]
        quantiles.append(estimates)
        diagnostics.append(details)
    return HourlySpeeds(tuple(quantiles), tuple(diagnostics))
