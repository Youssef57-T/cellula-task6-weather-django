from django.test import SimpleTestCase

from forecast.forms import ForecastForm
from forecast.services import get_forecaster

VALID = {"date": "2025-12-25", "hour": "6", "model": "SARIMA", "horizon": "24"}


def make(**changes):
    data = {**VALID, **changes}
    return ForecastForm({k: v for k, v in data.items() if v is not None}, forecaster=get_forecaster())


class ForecastFormTests(SimpleTestCase):
    def test_valid_input_builds_the_origin(self):
        form = make()
        self.assertTrue(form.is_valid(), form.errors)
        args = form.forecast_args()
        self.assertEqual(str(args["origin"]), "2025-12-25 06:00:00")
        self.assertIsNone(args["last_temp"])

    def test_current_temperature_is_optional_and_accepted(self):
        form = make(current_temp="15.5")
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.forecast_args()["last_temp"], 15.5)

    def test_missing_fields(self):
        form = make(date="", hour=None, model=None, horizon="")
        self.assertFalse(form.is_valid())
        self.assertEqual(set(form.errors), {"date", "hour", "model", "horizon"})
        self.assertIn("Choose the day", form.errors["date"][0])

    def test_non_numeric_text(self):
        for bad in ("warm", "12,5", "nan", "inf"):
            form = make(current_temp=bad)
            self.assertFalse(form.is_valid(), bad)
            self.assertIn("as a number", form.errors["current_temp"][0])
        self.assertIn("whole number", make(horizon="six").errors["horizon"][0])
        self.assertIn("date like", make(date="yesterday").errors["date"][0])

    def test_out_of_range_values(self):
        self.assertIn("at least -5", make(current_temp="-12").errors["current_temp"][0])
        self.assertIn("at most 50", make(current_temp="61").errors["current_temp"][0])
        self.assertIn("at most 24", make(horizon="48").errors["horizon"][0])
        self.assertIn("at least 1", make(horizon="0").errors["horizon"][0])
        self.assertIn("between 00:00 and 23:00", make(hour="24").errors["hour"][0])
        self.assertIn("one of the listed", make(model="Prophet").errors["model"][0])

    def test_start_outside_the_allowed_range(self):
        for date in ("2001-01-10", "2027-01-01", "1999-05-01"):
            form = make(date=date, hour="0")
            self.assertFalse(form.is_valid(), date)
            self.assertIn("Pick a start between", form.errors["date"][0])
        # any 2026 hour is allowed (forecast chained from the end of the data)
        self.assertTrue(make(date="2026-01-01", hour="1").is_valid())
        self.assertTrue(make(date="2026-07-15", hour="12").is_valid())
        self.assertTrue(make(date="2026-12-31", hour="23").is_valid())

    def test_jump_check_after_the_data_uses_the_model_estimate(self):
        estimate = get_forecaster().value_at("2026-03-01 04:00", "SARIMA")
        self.assertTrue(make(date="2026-03-01", current_temp=f"{estimate + 2:.1f}").is_valid())
        self.assertIn("largest one-hour change",
                      make(date="2026-03-01", current_temp=f"{estimate + 20:.1f}").errors["current_temp"][0])

    def test_impossible_jump_from_the_previous_hour(self):
        # 25 Dec 2025 04:00 was about 9.6 °C, so 45 °C at 05:00 is a jump no Cairo hour has ever had
        form = make(current_temp="45")
        self.assertFalse(form.is_valid())
        self.assertIn("largest one-hour change", form.errors["current_temp"][0])

    def test_invalid_fields_get_bootstrap_class(self):
        form = make(current_temp="abc")
        form.is_valid()
        form = ForecastForm(form.data, forecaster=get_forecaster())
        form.full_clean()
        self.assertIn("is-invalid", str(form["current_temp"]))
        self.assertNotIn("is-invalid", str(form["model"]))
