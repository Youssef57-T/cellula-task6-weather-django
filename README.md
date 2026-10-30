# Task 6: Cairo Temperature Forecast with Django

A Django web app around the **Task 5 forecasting models** (hourly air temperature in Cairo, NASA POWER data 2001–2025). The user picks when the forecast starts (any hour from 2001 to the end of 2026), can type the latest temperature reading, chooses a model and a horizon, and gets the next 1–24 hours with a 95 % range and a chart. For any start up to the end of 2025 the forecast is drawn next to what was actually recorded.

## Run it locally

Python 3.12 is recommended (3.10+ works).

```bash
cd Task6_Weather_Django
pip install -r requirements.txt
python manage.py runserver
```

Open http://127.0.0.1:8000/. Nothing else is needed: the trained models (`models/forecaster.joblib`, 471 kB) and the hourly history (`data/cairo_temperature_hourly.npz`, 520 kB) are in the repository, and the app uses no database.

Run the tests (form validation, page, JSON API and cache):

```bash
python manage.py test forecast
```

The tests that compare the app with Task 5's saved predictions need a `reference/` folder that is not in this repository, so they are skipped here.

## Using the app

| Input | What it does | Validation |
|---|---|---|
| Start date + start hour | First hour to forecast. The models read the temperatures before it. | Required; 26 Feb 2001 00:00 to 31 Dec 2026 23:00 (8 weeks of history are needed before the start) |
| Temperature in the hour before (optional) | Replaces the recorded value of the last hour, e.g. with your own reading. The form shows the recorded value as you change the date. | A number from −5 to 50 °C (Task 5's plausible range for Cairo), and no more than a 7.4 °C change from the hour before (the largest one-hour change in 25 years of data; in 2026 the hour before is the model's estimate) |
| Model | SARIMA, LSTM, Simple RNN, ARIMA, seasonal naive, persistence | Must be one of the list |
| Hours to forecast | 1–24 | Whole number from 1 to 24 |

Each invalid field is outlined in red with its own message under it, and the result panel asks you to check the highlighted fields. Empty fields, non-numeric text (`warm`, `12,5`, `nan`), out-of-range numbers and impossible jumps are all tested in `forecast/tests/test_forms.py`.

The result panel shows the first forecast hour as a large number. The panel's colour and icon follow the temperature (cold < 10 °C, cool, mild, warm, hot ≥ 33 °C). Below it are the low, the high, the model's typical error from Task 5, and the MAE against the recorded values when they exist. The chart shows the 48 recorded input hours, the forecast with its 95 % band, and the recorded values. An hourly table and a JSON link are at the bottom.

## The models: where they come from and how they match Task 5

Task 5 trained the models in its notebook but did not save them. An export script (kept with the Task 5 work, outside this repository) repeats the notebook's steps with the same data, windows, seeds and hyper-parameters, and writes one joblib bundle:

- **SARIMA(2,1,2)(1,1,1,24)** and **ARIMA(5,1,1)**: parameters fitted on the 8 weeks before the Task 5 test set (16 Oct – 10 Dec 2025).
- **LSTM (64→32)** and **Simple RNN (32)**: trained on 2022 → test start with Task 5's PyTorch code and seed. Only the weights and the MinMax scaler are saved. The app runs the forward pass in NumPy, so the server does not need PyTorch.
- Each model's per-hour RMSE on the Task 5 test set, used for the ranges of models without their own interval.

The bundle is loaded **once**, when Django starts (`ForecastConfig.ready`), and shared by all requests. At each request:

- **SARIMA/ARIMA:** the saved parameters are applied by the Kalman filter to the 8 weeks before the start, and `get_forecast` returns the mean and the model's own 95 % interval.
- **LSTM/RNN:** the networks forecast recursively from the 48 hours before the start, like Task 5.

`Task6.ipynb` checks that the app's predictions match the model's. It was executed against Task 5's saved test predictions, which are not part of this repository, and its outputs are saved:

| Check | Result |
|---|---|
| LSTM / Simple RNN day-ahead forecasts on the 21 Task 5 test days vs Task 5's saved predictions | max difference 0.0005 °C (the CSV has 3 decimals); MAE 1.12 / 1.55 °C, same as Task 5 |
| App's NumPy networks vs PyTorch with the same weights | max difference 4·10⁻⁵ °C |
| Seasonal naive, persistence | identical |
| SARIMA next-hour forecasts, 504 test hours | max difference 0.0009 °C; MAE 0.232 °C, same as Task 5 |
| SARIMA re-fitted from scratch with Task 5's settings vs the bundle | identical parameters, 24-h forecast and 95 % interval |
| JSON endpoint vs direct model call, every model | equal up to the JSON's 0.001 rounding |
| SARIMA day-ahead MAE | app 0.97 °C vs 0.95 °C in Task 5 |

The last row differs on purpose. Task 5's walk-forward test **re-estimated** SARIMA every day. The app keeps the parameters estimated once, as the brief asks ("load the model once"), because re-estimating takes several seconds per request. That costs 0.02 °C of MAE.

`requirements-train.txt` lists the extra packages (PyTorch, Jupyter) for re-creating the bundle and re-running the notebook.

## Starts in 2026: beyond the recorded data

The data ends on **31 Dec 2025 23:00**, but the calendar allows any start in 2026. For those starts each model first continues its own forecast hour by hour from the last recorded hour up to the start (the "chain"), then forecasts from it, as if its estimates were observations. The chains are computed once and stored in the bundle, so a 2026 request is as fast as any other.

The result panel says how many days the start is past the data. The 95 % range of SARIMA/ARIMA is counted from the last recorded hour, so it widens with the distance. The other models' ranges are their 24-hour Task 5 error, and the page says the real error is larger.

**Read 2026 forecasts with care.** Task 5 tested the models up to 21 days ahead (SARIMA: 1.5 °C MAE). Months ahead they go wrong in different ways, and the page shows a red warning past 21 days:

| Day | Cairo 2001–25 monthly mean | SARIMA | ARIMA | LSTM | Simple RNN |
|---|---|---|---|---|---|
| 7 Jan 2026 | 12.3 °C | 7.8 – 19.5 | 14.1 flat | 8.7 – 22.6 | 13.9 – 22.9 |
| 15 Feb 2026 | 13.6 °C | 3.5 – 15.2 | 14.1 flat | 15.6 – 30.2 | 14.5 – 23.8 |
| 15 Jul 2026 | 30.2 °C | −13.1 – −1.4 | 14.1 flat | 22.2 – 37.8 | 14.5 – 23.8 |

SARIMA keeps extending December's cooling trend, ARIMA settles on one value, and the networks repeat a fixed daily pattern.

## Bonus features

- **Model dropdown:** six models, with Task 5's scores shown next to the form.
- **Chart:** recorded inputs, forecast, 95 % band, and the recorded actuals (Chart.js).
- **JSON API:** `GET /api/forecast/?date=2025-12-25&hour=0&model=SARIMA&horizon=24[&current_temp=14]` returns the same forecast as the page, or HTTP 400 with the field errors.
- **Cache:** identical inputs return the stored result (Django cache; about 0.01 s instead of 0.5–2 s for SARIMA). The result shows a "cached" badge.

## Project layout

```
Task6_Weather_Django/
├── manage.py
├── weather_site/            Django project: settings (env-based), urls, wsgi
├── forecast/                the app
│   ├── forecaster.py        ML logic only (no Django): loads the bundle, runs each model
│   ├── services.py          one shared Forecaster per process + prediction cache
│   ├── forms.py             ForecastForm: fields, ranges, error messages, cross-field checks
│   ├── views.py             page, JSON API, recorded-value lookup
│   ├── templates/forecast/  base.html, index.html (Bootstrap 5)
│   ├── static/forecast/     style.css, app.js (chart, slider, hints)
│   └── tests/               forms, views/API, agreement with Task 5
├── models/forecaster.joblib trained models
├── data/                    hourly temperature history used as model input
├── Task6.ipynb              app vs model check, executed
└── render.yaml, Procfile    deployment
```

## Deploying (optional)

The settings read everything from environment variables:

| Variable | Production value |
|---|---|
| `DJANGO_DEBUG` | `0` (debug off) |
| `DJANGO_SECRET_KEY` | a long random string (required when debug is off) |
| `DJANGO_ALLOWED_HOSTS` | your domain, comma-separated (Render and Railway domains are added automatically) |

With debug off, WhiteNoise serves the CSS and JavaScript collected by `collectstatic`.

- **Render:** *New → Blueprint*, choose the repository; `render.yaml` sets the build (`pip install`, `collectstatic`), the start command (gunicorn) and a generated secret key.
- **Railway:** deploy the repository; the `Procfile` runs `collectstatic` and gunicorn. Set `DJANGO_DEBUG=0` and `DJANGO_SECRET_KEY` in the service variables.

The model files are small (about 1 MB together), so they are committed with the code.

## Notes

- Times are NASA POWER **local solar time** (about UTC+2 h 05 min), without daylight saving, as in Tasks 4 and 5.
- The Task 5 models are **univariate**: they forecast temperature from past temperature only. Humidity and wind are not model inputs, so the form does not ask for them.
- The ranges of LSTM, RNN and the baselines are ±1.96 × the model's Task 5 test RMSE at the same forecast hour (measured on December mornings to evenings). They describe a winter month and are an approximation in summer.
