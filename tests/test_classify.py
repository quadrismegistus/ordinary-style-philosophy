"""Tests for classification logic with synthetic data."""
import numpy as np
import pandas as pd

from osp.classify import classify_data


class TestClassifyData:
    def _make_synthetic_data(self, n=200, seed=42):
        """Create a separable two-class dataset."""
        rng = np.random.RandomState(seed)
        n_half = n // 2
        # Class A: features centered around 1
        X_a = rng.normal(loc=1.0, scale=0.5, size=(n_half, 5))
        # Class B: features centered around -1
        X_b = rng.normal(loc=-1.0, scale=0.5, size=(n_half, 5))
        X = np.vstack([X_a, X_b])
        y = ["A"] * n_half + ["B"] * n_half

        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
        df["_target"] = y
        df.index = [f"id_{i}" for i in range(n)]
        return df

    def test_returns_three_items(self):
        data = self._make_synthetic_data()
        results_df, weights_df, model = classify_data(data, verbose=False)
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(weights_df, pd.DataFrame)

    def test_accuracy_on_separable_data(self):
        data = self._make_synthetic_data()
        results_df, weights_df, model = classify_data(data, verbose=False)
        accuracy = results_df["accuracy"].iloc[0]
        # Clearly separable data should give high accuracy
        assert accuracy > 0.8

    def test_predictions_have_expected_columns(self):
        data = self._make_synthetic_data()
        results_df, weights_df, model = classify_data(data, verbose=False)
        assert "true_label" in results_df.columns
        assert "pred_label" in results_df.columns
        assert "confidence" in results_df.columns
        assert "correct" in results_df.columns

    def test_weights_have_features(self):
        data = self._make_synthetic_data()
        results_df, weights_df, model = classify_data(data, verbose=False)
        assert "feature" in weights_df.columns
        assert "weight" in weights_df.columns
        assert len(weights_df) == 5  # 5 features

    def test_balanced_sampling(self):
        data = self._make_synthetic_data(n=200)
        results_df, _, _ = classify_data(
            data, verbose=False, balance=True, sample_size=50
        )
        # Each class should have 50 samples
        assert len(results_df) == 100
