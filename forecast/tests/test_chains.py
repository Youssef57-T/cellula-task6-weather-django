"""Forecasts that start after the recorded data (2026), built on each model's chained forecast."""
import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from forecast.forecaster import MODELS
from forecast.services import get_forecaster


class ChainTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fc = get_forecaster()

    def test_range_covers_2026(self):
        self.assertEqual(self.fc.last_origin, pd.Timestamp("2026-12-31 23:00"))

    def test_chain_continues_the_forecast_from_the_data_end(self):
        # the first hour after the data: a normal forecast, equal to the start of each chain
        start = self.fc.data_end + pd.Timedelta(hours=1)
        for model in MODELS:
            f = self.fc.forecast(start, model, 24)
            np.testing.assert_allclose(f.mean, self.fc.chains[model]["mean"][:24], atol=1e-6, err_msg=model)
            self.assertEqual(f.gap_hours, 0)

    def test_later_start_is_a_slice_of_the_chain(self):
        origin = pd.Timestamp("2026-03-01 06:00")
        gap = int((origin - self.fc.data_end) / pd.Timedelta(hours=1)) - 1
        for model in MODELS:
            f = self.fc.forecast(origin, model, 12)
            np.testing.assert_allclose(f.mean, self.fc.chains[model]["mean"][gap:gap + 12], err_msg=model)
            self.assertEqual(f.gap_hours, gap)
            self.assertIsNone(f.last_recorded)
            self.assertFalse(f.recorded_mask.any())
            self.assertTrue(f.actual.isna().all())

    def test_sarima_chain_matches_a_direct_long_forecast(self):
        past = self.fc.history.iloc[-self.fc.window_hours:]
        mean, lower, _ = self.fc._statespace("SARIMA", past, 300)
        np.testing.assert_allclose(self.fc.chains["SARIMA"]["mean"][:300], mean, atol=1e-8)
        np.testing.assert_allclose(self.fc.chains["SARIMA"]["half"][:300], mean - lower, atol=1e-5)

    def test_history_mixes_recordings_and_estimates_near_the_data_end(self):
        f = self.fc.forecast("2026-01-01 12:00", "LSTM", 6)
        self.assertEqual(int(f.recorded_mask.sum()), 36)       # 30 Dec 12:00 - 31 Dec 23:00
        self.assertEqual(int((~f.recorded_mask).sum()), 12)    # 1 Jan 00:00 - 11:00, the model's estimates

    def test_override_after_the_data(self):
        origin = "2026-02-10 08:00"
        base = self.fc.forecast(origin, "SARIMA", 6)
        warmer = self.fc.forecast(origin, "SARIMA", 6, last_temp=base.last_used + 3)
        self.assertTrue(warmer.overridden)
        self.assertGreater(warmer.mean.iloc[0], base.mean.iloc[0] + 1)
        # the range still counts from the last recording
        np.testing.assert_allclose(warmer.upper - warmer.mean, base.upper - base.mean)

    def test_ranges_widen_with_distance(self):
        near = self.fc.forecast("2026-01-03 00:00", "SARIMA", 1)
        far = self.fc.forecast("2026-06-01 00:00", "SARIMA", 1)
        self.assertGreater((far.upper - far.lower).iloc[0], 10 * (near.upper - near.lower).iloc[0])
        for model in MODELS:
            f = self.fc.forecast("2026-09-09 09:00", model, 24)
            self.assertTrue(((f.lower <= f.mean) & (f.mean <= f.upper)).all(), model)
