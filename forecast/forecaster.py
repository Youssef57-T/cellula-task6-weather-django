"""Inference for the Task 5 Cairo temperature models. Plain Python, no Django imports.

The bundle written by ``scripts/export_models.py`` holds the fitted SARIMA/ARIMA
parameters and the Simple RNN/LSTM weights. Forecasts reproduce the notebook:

* SARIMA/ARIMA: the saved parameters are applied (Kalman filter, no re-estimation)
  to the 8 weeks before the forecast start, then ``get_forecast`` gives the mean
  and its 95 % interval. At the Task 5 test start this is the notebook's model exactly.
* Simple RNN/LSTM: the 48 hours before the start are min-max scaled and the
  network predicts recursively, feeding each prediction back in (Task 5 cell 67).
  The forward pass is written in NumPy so the app does not need PyTorch.
* Seasonal naive / persistence: same hour yesterday / last hour repeated.

Models without their own interval get ±1.96 × the RMSE of their Task 5
day-ahead test errors at the same forecast step.

Starts after the recorded data (2026): each model's own forecast from the last
recorded hour (the "chain", precomputed in the bundle) stands in for the missing
hours. Errors grow with the distance from the data; Task 5 tested up to 21 days.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# numpy 2.x notice raised inside pandas date arithmetic (silenced in Task 5 too)
warnings.filterwarnings("ignore", message=".*'generic' unit for NumPy timedelta", category=DeprecationWarning)

MODELS = {
    "SARIMA": "SARIMA(2,1,2)(1,1,1,24)",
    "LSTM": "LSTM (64 → 32)",
    "Simple RNN": "Simple RNN (32)",
    "ARIMA": "ARIMA(5,1,1)",
    "Seasonal naive": "Seasonal naive (same hour yesterday)",
    "Persistence": "Persistence (last hour repeated)",
}
STATESPACE = ("SARIMA", "ARIMA")
MAX_HORIZON = 24
TESTED_DAYS = 21  # longest forecast distance evaluated in Task 5
Z95 = 1.959964


@dataclass
class Forecast:
    model: str
    origin: pd.Timestamp
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    interval: str                     # "model" (SARIMA/ARIMA) or "task5_errors"
    history: pd.Series                # the 48 hours before the origin, as used by the model
    recorded_mask: pd.Series          # True where a history hour is a recording, False where it is the model's chain
    actual: pd.Series                 # recorded values for the forecast hours (NaN after the data ends)
    last_recorded: float | None       # None when the hour before the start is after the data
    last_used: float
    overridden: bool
    data_end: pd.Timestamp
    gap_hours: int                    # hours between the last recording and the start (0 inside the data)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        def num(v):
            return None if v is None or pd.isna(v) else round(float(v), 3)

        return {
            "model": self.model,
            "model_label": MODELS[self.model],
            "origin": self.origin.isoformat(),
            "horizon": len(self.mean),
            "interval": self.interval,
            "data_end": self.data_end.isoformat(),
            "gap_hours": self.gap_hours,
            "gap_days": round(self.gap_hours / 24, 1),
            "beyond_tested": self.gap_hours > TESTED_DAYS * 24,
            "last_hour": {"time": self.history.index[-1].isoformat(), "recorded": num(self.last_recorded),
                          "used": num(self.last_used), "overridden": self.overridden},
            "forecast": [{"time": t.isoformat(), "temperature": num(m), "lower": num(lo), "upper": num(hi),
                          "actual": num(a)}
                         for t, m, lo, hi, a in zip(self.mean.index, self.mean, self.lower, self.upper, self.actual)],
            "history": [{"time": t.isoformat(), "temperature": num(v), "recorded": bool(r)}
                        for (t, v), r in zip(self.history.items(), self.recorded_mask)],
            **self.extra,
        }


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm_layer(x, w, prefix):
    """PyTorch nn.LSTM (batch_first, one layer) on a (T, n_in) sequence; gate order i, f, g, o."""
    w_hh = w[f"{prefix}.weight_hh_l0"]
    hidden = w_hh.shape[1]
    gates_x = x @ w[f"{prefix}.weight_ih_l0"].T + w[f"{prefix}.bias_ih_l0"] + w[f"{prefix}.bias_hh_l0"]
    h, c = np.zeros(hidden), np.zeros(hidden)
    out = np.empty((len(x), hidden))
    for t, gx in enumerate(gates_x):
        z = gx + w_hh @ h
        i, f, g, o = _sigmoid(z[:hidden]), _sigmoid(z[hidden:2 * hidden]), np.tanh(z[2 * hidden:3 * hidden]), \
            _sigmoid(z[3 * hidden:])
        c = f * c + i * g
        h = o * np.tanh(c)
        out[t] = h
    return out


def _rnn_layer(x, w, prefix):
    """PyTorch nn.RNN (tanh, batch_first, one layer) on a (T, n_in) sequence."""
    w_hh = w[f"{prefix}.weight_hh_l0"]
    hx = x @ w[f"{prefix}.weight_ih_l0"].T + w[f"{prefix}.bias_ih_l0"] + w[f"{prefix}.bias_hh_l0"]
    h = np.zeros(w_hh.shape[0])
    for gx in hx:
        h = np.tanh(gx + w_hh @ h)
    return h


def _network_step(name, w, window):
    """One forward pass of Task 5's network on a scaled window of shape (n_steps,); dropout is off at inference."""
    x = window[:, None]
    if name == "LSTM":
        h = _lstm_layer(_lstm_layer(x, w, "lstm1"), w, "lstm2")[-1]
    else:
        h = _rnn_layer(x, w, "rnn")
    return float(w["out.weight"][0] @ h + w["out.bias"][0])


class Forecaster:
    """Loads the model bundle and the hourly history once; ``forecast`` is called per request."""

    def __init__(self, model_path, history_path):
        self.bundle = joblib.load(Path(model_path))
        hist = np.load(Path(history_path))
        start = pd.Timestamp(str(hist["start"]))
        values = hist["temperature"]
        self.history = pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), name="temperature")
        self.window_hours = self.bundle["sarima"]["window_hours"]
        self.n_steps = self.bundle["networks"]["n_steps"]
        self.data_end = self.history.index[-1]
        # Earliest start that leaves a full SARIMA window; latest is the last hour the chains reach
        self.first_origin = self.history.index[0] + pd.Timedelta(hours=self.window_hours)
        self.chains = self.bundle.get("chains", {})
        if self.chains:
            n = len(self.chains["SARIMA"]["mean"])
            self.last_origin = self.data_end + pd.Timedelta(hours=n - MAX_HORIZON + 1)
        else:
            self.last_origin = self.data_end + pd.Timedelta(hours=1)
        self.max_hourly_change = self.bundle["max_hourly_change"]

    def recorded(self, ts):
        ts = pd.Timestamp(ts)
        return float(self.history[ts]) if ts in self.history.index else None

    def value_at(self, ts, model="SARIMA"):
        """Recorded value, or the model's own chained forecast for an hour after the data."""
        ts = pd.Timestamp(ts)
        if ts <= self.data_end:
            return self.recorded(ts)
        lead = int((ts - self.data_end) / pd.Timedelta(hours=1))
        return float(self.chains[model]["mean"][lead - 1])

    def task5_scores(self):
        return self.bundle["task5_day_ahead"]

    def forecast(self, origin, model="SARIMA", horizon=MAX_HORIZON, last_temp=None):
        origin = pd.Timestamp(origin)
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}")
        if not 1 <= horizon <= MAX_HORIZON:
            raise ValueError(f"horizon must be 1-{MAX_HORIZON} hours")
        if not self.first_origin <= origin <= self.last_origin or origin.minute or origin.second:
            raise ValueError(f"origin must be a whole hour between {self.first_origin} and {self.last_origin}")

        gap = int((origin - self.data_end) / pd.Timedelta(hours=1)) - 1   # hours of chain before the origin
        series = self.history
        if gap > 0:
            chain = pd.Series(self.chains[model]["mean"][:gap],
                              index=pd.date_range(self.data_end + pd.Timedelta(hours=1), periods=gap, freq="h"))
            series = pd.concat([self.history.iloc[-self.window_hours:], chain])
        past = series.loc[:origin - pd.Timedelta(hours=1)].iloc[-self.window_hours:].copy()
        recorded_mask = pd.Series(past.index <= self.data_end, index=past.index)
        recorded = self.recorded(past.index[-1])
        if last_temp is not None:
            past.iloc[-1] = float(last_temp)
        index = pd.date_range(origin, periods=horizon, freq="h")

        if gap > 0 and last_temp is None:
            # nothing new since the last recording: the forecast is the model's chain
            ch = self.chains[model]
            mean = ch["mean"][gap:gap + horizon]
        else:
            mean, lower, upper = self._predict(model, past, horizon)

        if model in STATESPACE:
            if gap > 0:
                # uncertainty is counted from the last recording, not from the start
                half = self.chains[model]["half"][gap:gap + horizon]
                lower, upper = mean - half, mean + half
            interval = "model"
        else:
            # Task 5 measured errors up to 24 h ahead; further from the data the last value is reused
            steps = np.minimum(np.arange(gap + 1, gap + 1 + horizon), MAX_HORIZON) - 1 if gap > 0 \
                else np.arange(horizon)
            half = Z95 * self.bundle["error_rmse_by_step"][model][steps]
            lower, upper = mean - half, mean + half
            interval = "task5_errors"

        return Forecast(model=model, origin=origin, mean=pd.Series(mean, index), lower=pd.Series(lower, index),
                        upper=pd.Series(upper, index), interval=interval, history=past.iloc[-48:],
                        recorded_mask=recorded_mask.iloc[-48:], actual=self.history.reindex(index),
                        last_recorded=recorded, last_used=float(past.iloc[-1]), overridden=last_temp is not None,
                        data_end=self.data_end, gap_hours=max(gap, 0))

    def _predict(self, model, past, horizon):
        """Mean (and, for SARIMA/ARIMA, the 95 % bounds) of a forecast from the hours in ``past``."""
        if model in STATESPACE:
            return self._statespace(model, past, horizon)
        if model in ("LSTM", "Simple RNN"):
            mean = self._recursive(model, past.iloc[-self.n_steps:].to_numpy(), horizon)
        elif model == "Seasonal naive":
            mean = np.resize(past.iloc[-24:].to_numpy(), horizon)
        else:
            mean = np.repeat(past.iloc[-1], horizon)
        return mean, None, None

    def _statespace(self, model, past, horizon):
        spec = self.bundle["sarima"] if model == "SARIMA" else self.bundle["arima"]
        mod = SARIMAX(past, order=spec["order"], seasonal_order=spec.get("seasonal_order", (0, 0, 0, 0)),
                      enforce_stationarity=False, enforce_invertibility=False)
        fc = mod.filter(spec["params"]).get_forecast(horizon)
        ci = fc.conf_int(alpha=0.05).to_numpy()
        return fc.predicted_mean.to_numpy(), ci[:, 0], ci[:, 1]

    def _statespace_long(self, model, past, horizon, block=1000):
        """Same as ``_statespace`` for long horizons. ``get_forecast`` keeps a state covariance matrix per step
        (too much memory for a year), so the future hours are filtered as missing observations in blocks,
        each block continuing from the last state of the one before."""
        spec = self.bundle["sarima"] if model == "SARIMA" else self.bundle["arima"]
        mod = SARIMAX(past, order=spec["order"], seasonal_order=spec.get("seasonal_order", (0, 0, 0, 0)),
                      enforce_stationarity=False, enforce_invertibility=False)
        res, means, variances, start = mod.filter(spec["params"]), [], [], past.index[-1]
        for n in range(0, horizon, block):
            size = min(block, horizon - n)
            gap = pd.Series(np.nan, index=pd.date_range(start + pd.Timedelta(hours=1), periods=size, freq="h"))
            res, start = res.extend(gap), gap.index[-1]
            means.append(res.forecasts[0])
            variances.append(res.forecasts_error_cov[0, 0])
        mean = np.concatenate(means)
        half = Z95 * np.sqrt(np.concatenate(variances))
        return mean, mean - half, mean + half

    def _recursive(self, model, last_hours, horizon):
        net = self.bundle["networks"]
        lo, hi = net["scaler_min"], net["scaler_max"]
        window = (np.asarray(last_hours, float) - lo) / (hi - lo)
        out = []
        for _ in range(horizon):
            nxt = _network_step(model, net["weights"][model], window)
            out.append(nxt)
            window = np.append(window[1:], nxt)
        return np.asarray(out) * (hi - lo) + lo

    def build_chains(self, hours):
        """Each model's forecast for the ``hours`` after the last recording (stored in the bundle by the export)."""
        past = self.history.iloc[-self.window_hours:]
        chains = {}
        for model in MODELS:
            if model in STATESPACE:
                mean, lower, upper = self._statespace_long(model, past, hours)
            else:
                mean, lower, upper = self._predict(model, past, hours)
            chains[model] = {"mean": np.asarray(mean, np.float64),
                             "half": None if lower is None else np.asarray((upper - lower) / 2, np.float64)}
        return chains
