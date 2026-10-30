"""Glue between the views and the ML code: the shared Forecaster and the prediction cache."""
import hashlib
from functools import lru_cache

import numpy as np
from django.conf import settings
from django.core.cache import cache

from .forecaster import Forecaster


@lru_cache(maxsize=1)
def get_forecaster():
    """The single Forecaster of this process; built on first call (at startup, from ForecastConfig.ready)."""
    return Forecaster(settings.FORECAST_MODEL_PATH, settings.FORECAST_HISTORY_PATH)


def _summary(data, scores):
    hours = data["forecast"]
    temps = np.array([h["temperature"] for h in hours])
    low, high = int(temps.argmin()), int(temps.argmax())
    pairs = [(h["temperature"], h["actual"]) for h in hours if h["actual"] is not None]
    return {
        "next": hours[0],
        "low": hours[low],
        "high": hours[high],
        "mean": round(float(temps.mean()), 2),
        "task5_mae": round(scores[data["model"]]["MAE"], 2),
        "actual_hours": len(pairs),
        "mae_vs_actual": round(float(np.mean([abs(f - a) for f, a in pairs])), 2) if pairs else None,
    }


def run_forecast(origin, model, horizon, last_temp=None):
    """Forecast as a JSON-ready dict, plus whether it came from the cache."""
    raw = f"{model}|{origin.isoformat()}|{horizon}|{last_temp}"
    key = "forecast:v1:" + hashlib.sha1(raw.encode()).hexdigest()
    data = cache.get(key)
    if data is not None:
        return data, True
    forecaster = get_forecaster()
    data = forecaster.forecast(origin, model=model, horizon=horizon, last_temp=last_temp).to_dict()
    data["summary"] = _summary(data, forecaster.task5_scores())
    cache.set(key, data, settings.FORECAST_CACHE_SECONDS)
    return data, False
