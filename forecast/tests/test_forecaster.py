"""The app's forecasts against the predictions Task 5 saved (reference/task5_test_predictions.csv)."""
from pathlib import Path

import unittest

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from forecast.forecaster import MODELS
from forecast.services import get_forecaster

REFERENCE = Path(__file__).resolve().parents[2] / "reference" / "task5_test_predictions.csv"


@unittest.skipUnless(REFERENCE.exists(), "Task 5 reference predictions not present")
class Task5AgreementTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fc = get_forecaster()
        cls.ref = pd.read_csv(REFERENCE, index_col=0, parse_dates=True)
        cls.origins = cls.ref.index[::24]  # the 21 test midnights

    def day_ahead(self, model):
        return pd.concat([self.fc.forecast(o, model=model, horizon=24).mean for o in self.origins])

    def test_history_matches_task5_test_data(self):
        np.testing.assert_allclose(self.fc.history.reindex(self.ref.index), self.ref.actual, atol=1e-3)

    def test_networks_reproduce_task5_day_ahead_forecasts(self):
        for model in ("LSTM", "Simple RNN"):
            diff = (self.day_ahead(model) - self.ref[f"day_ahead_{model}"]).abs().max()
            self.assertLess(diff, 0.01, model)

    def test_baselines_reproduce_task5(self):
        for model in ("Seasonal naive", "Persistence"):
            np.testing.assert_allclose(self.day_ahead(model), self.ref[f"day_ahead_{model}"], atol=1e-3)

    def test_sarima_one_step_matches_task5(self):
        # Task 5 kept the parameters fixed and ran the filter over the test hours: same as the app at horizon 1
        origins = self.ref.index[::6]
        app = pd.Series([self.fc.forecast(o, model="SARIMA", horizon=1).mean.iloc[0] for o in origins], origins)
        self.assertLess((app - self.ref.loc[origins, "one_step_SARIMA"]).abs().max(), 0.01)

    def test_sarima_day_ahead_is_as_accurate_as_task5(self):
        # Task 5 re-estimated SARIMA every day (MAE 0.95 °C); the app keeps the saved parameters
        mae = (self.day_ahead("SARIMA") - self.ref.actual).abs().mean()
        self.assertLess(mae, 1.1)

    def test_override_changes_the_forecast(self):
        origin = self.origins[3]
        base = self.fc.forecast(origin, "SARIMA", 6)
        warmer = self.fc.forecast(origin, "SARIMA", 6, last_temp=base.last_recorded + 3)
        self.assertGreater(warmer.mean.iloc[0], base.mean.iloc[0] + 1)
        self.assertEqual(warmer.last_recorded, base.last_recorded)

    def test_intervals_contain_the_forecast(self):
        for model in MODELS:
            f = self.fc.forecast(self.origins[0], model, 24)
            self.assertTrue(((f.lower <= f.mean) & (f.mean <= f.upper)).all(), model)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            self.fc.forecast("2027-01-01 00:00", "SARIMA", 24)
        with self.assertRaises(ValueError):
            self.fc.forecast(self.origins[0], "SARIMA", 25)
        with self.assertRaises(ValueError):
            self.fc.forecast(self.origins[0], "Prophet", 24)
