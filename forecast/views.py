from datetime import timedelta

from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_GET

from . import services
from .forms import ForecastForm

# (upper bound °C, name, Bootstrap icon) for colour-coding the result panel
BANDS = [(10, "cold", "bi-snow2"), (18, "cool", "bi-cloud-sun"), (26, "mild", "bi-brightness-high"),
         (33, "warm", "bi-sun"), (float("inf"), "hot", "bi-thermometer-sun")]


def temperature_band(value):
    return next({"name": name, "icon": icon} for top, name, icon in BANDS if value < top)


def _form(request):
    """A bound form when the query string carries any form field, otherwise an empty one."""
    data = request.GET if set(ForecastForm.base_fields) & set(request.GET) else None
    return ForecastForm(data, forecaster=services.get_forecaster())


@require_GET
def index(request):
    form = _form(request)
    result = None
    if form.is_bound and form.is_valid():
        result, cached = services.run_forecast(**form.forecast_args())
        result = {**result, "cached": cached, "band": temperature_band(result["summary"]["next"]["temperature"])}
    scores = services.get_forecaster().task5_scores()
    return render(request, "forecast/index.html", {
        "form": form,
        "result": result,
        "scores": sorted(((m, s["MAE"], s["RMSE"]) for m, s in scores.items()), key=lambda r: r[1]),
        "api_url": f"{reverse('forecast:api')}?{request.GET.urlencode()}" if result else None,
    })


@require_GET
def api_forecast(request):
    """JSON version of the forecast, same query parameters as the page."""
    form = ForecastForm(request.GET, forecaster=services.get_forecaster())
    if not form.is_valid():
        return JsonResponse({"errors": {k: [e["message"] for e in v] for k, v in form.errors.get_json_data().items()}},
                            status=400)
    result, cached = services.run_forecast(**form.forecast_args())
    return JsonResponse({**result, "cached": cached})


@require_GET
def api_observation(request):
    """Recorded temperature in the hour before a start time; the form shows it as the default reading."""
    form = ForecastForm({**request.GET.dict(), "model": "SARIMA", "horizon": 1}, forecaster=services.get_forecaster())
    if not form.is_valid() or "origin" not in form.cleaned_data:
        return JsonResponse({"error": "invalid start"}, status=400)
    last = form.cleaned_data["origin"] - timedelta(hours=1)
    fc = services.get_forecaster()
    return JsonResponse({"time": last.isoformat(), "temperature": fc.recorded(last),
                         "data_end": fc.data_end.isoformat()})
