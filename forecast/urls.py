from django.urls import path

from . import views

app_name = "forecast"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/forecast/", views.api_forecast, name="api"),
    path("api/observation/", views.api_observation, name="observation"),
]
