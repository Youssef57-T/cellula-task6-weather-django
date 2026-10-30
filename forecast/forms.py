from datetime import datetime, timedelta

from django import forms

from .forecaster import MAX_HORIZON, MODELS

# Task 5 section 2.3: physically plausible temperature range for Cairo (°C)
TEMP_MIN, TEMP_MAX = -5, 50


def _fmt(ts):
    return f"{ts:%d %b %Y %H:%M}"


class ForecastForm(forms.Form):
    date = forms.DateField(
        label="Start date",
        widget=forms.DateInput(attrs={"type": "date"}),
        error_messages={"required": "Choose the day the forecast starts.",
                        "invalid": "Enter a date like 2025-12-25."},
    )
    hour = forms.TypedChoiceField(
        label="Start hour",
        choices=[(h, f"{h:02d}:00") for h in range(24)],
        coerce=int,
        initial=0,
        error_messages={"required": "Choose the first hour to forecast.",
                        "invalid_choice": "Choose an hour between 00:00 and 23:00."},
    )
    current_temp = forms.FloatField(
        label="Temperature in the hour before (°C)",
        required=False,
        min_value=TEMP_MIN,
        max_value=TEMP_MAX,
        widget=forms.NumberInput(attrs={"step": "0.1", "placeholder": "recorded value", "inputmode": "decimal"}),
        error_messages={"invalid": "Enter the temperature as a number, for example 14.5.",
                        "min_value": f"Enter at least {TEMP_MIN} °C, the lowest plausible value for Cairo.",
                        "max_value": f"Enter at most {TEMP_MAX} °C, the highest plausible value for Cairo."},
    )
    model = forms.ChoiceField(
        label="Model",
        choices=list(MODELS.items()),
        initial="SARIMA",
        error_messages={"required": "Choose a model.", "invalid_choice": "Choose one of the listed models."},
    )
    horizon = forms.IntegerField(
        label="Hours to forecast",
        min_value=1,
        max_value=MAX_HORIZON,
        initial=MAX_HORIZON,
        widget=forms.NumberInput(attrs={"type": "range", "step": 1}),
        error_messages={"required": "Choose how many hours to forecast.",
                        "invalid": f"Enter a whole number of hours from 1 to {MAX_HORIZON}.",
                        "min_value": "Forecast at least 1 hour.",
                        "max_value": f"The models forecast at most {MAX_HORIZON} hours ahead."},
    )

    def __init__(self, *args, forecaster, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecaster = forecaster
        self.first, self.last = forecaster.first_origin, forecaster.last_origin
        self.fields["date"].widget.attrs.update(min=f"{self.first:%Y-%m-%d}", max=f"{self.last:%Y-%m-%d}")
        self.fields["date"].initial = f"{self.last:%Y-%m-%d}"
        self.fields["date"].help_text = f"{self.first:%d %b %Y} to {self.last:%d %b %Y}"
        self.fields["current_temp"].help_text = "Leave empty to use the recorded value."
        for name, field in self.fields.items():
            css = "form-select" if isinstance(field.widget, forms.Select) else \
                "form-range" if name == "horizon" else "form-control"
            if self.is_bound and name in self.errors:
                css += " is-invalid"
            field.widget.attrs["class"] = css

    def clean(self):
        data = super().clean()
        if data.get("date") is None or data.get("hour") is None:
            return data
        origin = datetime.combine(data["date"], datetime.min.time()) + timedelta(hours=data["hour"])
        if not self.first <= origin <= self.last:
            self.add_error("date", f"Pick a start between {_fmt(self.first)} and {_fmt(self.last)} "
                                   f"(the models need 8 weeks of history before the start).")
            return data
        data["origin"] = origin

        temp = data.get("current_temp")
        if temp is not None and data.get("model"):
            # after the data, the hour before is the model's own estimate
            before = self.forecaster.value_at(origin - timedelta(hours=2), data["model"])
            jump = abs(temp - before)
            if jump > self.forecaster.max_hourly_change:
                self.add_error("current_temp",
                               f"That is a {jump:.1f} °C change from the hour before ({before:.1f} °C). "
                               f"The largest one-hour change in 25 years of Cairo data is "
                               f"{self.forecaster.max_hourly_change:.1f} °C.")
        return data

    def forecast_args(self):
        d = self.cleaned_data
        return {"origin": d["origin"], "model": d["model"], "horizon": d["horizon"], "last_temp": d.get("current_temp")}
