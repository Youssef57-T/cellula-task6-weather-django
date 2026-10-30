from django.urls import include, path

urlpatterns = [
    path("", include("forecast.urls")),
]
