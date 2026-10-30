from django.core.cache import cache
from django.test import SimpleTestCase
from django.urls import reverse

QUERY = {"date": "2025-12-25", "hour": "0", "model": "SARIMA", "horizon": "24"}


class PageTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_empty_page_shows_form_and_no_result(self):
        r = self.client.get(reverse("forecast:index"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Cairo Temperature Forecast")
        self.assertContains(r, 'id="forecast-form"')
        self.assertContains(r, "Pick a start time")
        self.assertNotContains(r, 'id="result"')
        self.assertNotContains(r, "is-invalid")

    def test_valid_submission_renders_result_on_same_page(self):
        r = self.client.get(reverse("forecast:index"), QUERY)
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'id="result"')
        self.assertContains(r, "Forecast for 25 Dec 2025, 00:00")
        self.assertContains(r, 'id="forecast-data"')
        self.assertContains(r, "vs recorded")
        result = r.context["result"]
        self.assertEqual(len(result["forecast"]), 24)
        self.assertFalse(result["cached"])
        self.assertIn(result["band"]["name"], {"cold", "cool", "mild", "warm", "hot"})
        # the form keeps the submitted values
        self.assertContains(r, 'value="2025-12-25"')

    def test_second_identical_request_comes_from_the_cache(self):
        self.client.get(reverse("forecast:index"), QUERY)
        r = self.client.get(reverse("forecast:index"), QUERY)
        self.assertTrue(r.context["result"]["cached"])
        self.assertContains(r, "cached")

    def test_invalid_submission_shows_inline_errors(self):
        r = self.client.get(reverse("forecast:index"), {**QUERY, "current_temp": "hot", "horizon": "99"})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.context["result"])
        self.assertContains(r, "is-invalid", count=2)
        self.assertContains(r, "invalid-feedback", count=2)
        self.assertContains(r, "Check the highlighted fields")

    def test_future_start_has_no_recorded_values(self):
        r = self.client.get(reverse("forecast:index"), {**QUERY, "date": "2026-01-01", "model": "LSTM"})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.context["result"]["summary"]["mae_vs_actual"])
        self.assertContains(r, "future: no data yet")
        self.assertNotContains(r, "gap-alert")  # 1 Jan 00:00 directly follows the data

    def test_2026_start_shows_distance_warning(self):
        r = self.client.get(reverse("forecast:index"), {**QUERY, "date": "2026-01-10"})
        self.assertContains(r, "alert-warning")
        self.assertContains(r, "9 days after the last recorded hour")
        self.assertContains(r, "No recording for")
        r = self.client.get(reverse("forecast:index"), {**QUERY, "date": "2026-07-15", "model": "Simple RNN"})
        self.assertContains(r, "alert-danger")
        self.assertContains(r, "not reliable")
        self.assertTrue(r.context["result"]["beyond_tested"])

    def test_every_model_renders(self):
        for model in ("SARIMA", "LSTM", "Simple RNN", "ARIMA", "Seasonal naive", "Persistence"):
            r = self.client.get(reverse("forecast:index"), {**QUERY, "model": model, "horizon": "6"})
            self.assertEqual(r.status_code, 200, model)
            self.assertEqual(len(r.context["result"]["forecast"]), 6, model)

    def test_post_is_not_allowed(self):
        self.assertEqual(self.client.post(reverse("forecast:index"), QUERY).status_code, 405)


class ApiTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_json_forecast(self):
        r = self.client.get(reverse("forecast:api"), {**QUERY, "current_temp": "12"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["model"], "SARIMA")
        self.assertEqual(data["origin"], "2025-12-25T00:00:00")
        self.assertEqual(len(data["forecast"]), 24)
        self.assertEqual(len(data["history"]), 48)
        self.assertTrue(data["last_hour"]["overridden"])
        self.assertEqual(data["last_hour"]["used"], 12.0)
        for hour in data["forecast"]:
            self.assertLessEqual(hour["lower"], hour["temperature"])
            self.assertLessEqual(hour["temperature"], hour["upper"])

    def test_json_errors(self):
        r = self.client.get(reverse("forecast:api"), {"date": "2030-01-01", "hour": "0", "model": "SARIMA"})
        self.assertEqual(r.status_code, 400)
        errors = r.json()["errors"]
        self.assertIn("date", errors)
        self.assertIn("horizon", errors)

    def test_page_and_api_agree(self):
        page = self.client.get(reverse("forecast:index"), QUERY).context["result"]
        api = self.client.get(reverse("forecast:api"), QUERY).json()
        self.assertEqual(page["forecast"], api["forecast"])

    def test_observation(self):
        r = self.client.get(reverse("forecast:observation"), {"date": "2025-12-25", "hour": "0"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["time"], "2025-12-24T23:00:00")
        self.assertEqual(self.client.get(reverse("forecast:observation"), {"date": "x"}).status_code, 400)
        r = self.client.get(reverse("forecast:observation"), {"date": "2026-05-01", "hour": "0"})
        self.assertIsNone(r.json()["temperature"])
