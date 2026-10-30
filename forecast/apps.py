from django.apps import AppConfig


class ForecastConfig(AppConfig):
    name = "forecast"
    verbose_name = "Cairo temperature forecast"

    def ready(self):
        from .services import get_forecaster

        get_forecaster()  # load the model bundle once, when the server starts
