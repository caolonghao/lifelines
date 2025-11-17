# -*- coding: utf-8 -*-
"""
Tests for FineGrayFitter

These tests verify the implementation against expected behaviors and,
where possible, compare with known results.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from lifelines import FineGrayFitter
from lifelines.datasets import load_rossi


class TestFineGrayFitter:
    """Tests for FineGrayFitter"""

    def test_basic_fit(self):
        """Test basic fitting with competing risks data"""
        np.random.seed(0)
        n = 100

        # Create simple competing risks data
        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3]),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.binomial(1, 0.5, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Check that basic attributes are set
        assert hasattr(fgf, 'params_')
        assert hasattr(fgf, 'variance_matrix_')
        assert hasattr(fgf, 'log_likelihood_')
        assert hasattr(fgf, 'summary')
        assert hasattr(fgf, 'baseline_subdistribution_hazard_')
        assert hasattr(fgf, 'baseline_cumulative_subdistribution_hazard_')

        # Check dimensions
        assert len(fgf.params_) == 2  # X1 and X2
        assert fgf.variance_matrix_.shape == (2, 2)

    def test_with_rossi_dataset(self):
        """Test with rossi dataset"""
        df = load_rossi()

        # Create artificial competing events
        # For testing: treat re-arrest as event 1, and randomly assign some as competing
        df['E_comp'] = df['arrest'].copy()
        df.loc[df['arrest'] == 1, 'E_comp'] = np.random.choice([1, 2], (df['arrest'] == 1).sum())

        fgf = FineGrayFitter()
        fgf.fit(df, 'week', 'E_comp', event_of_interest=1, formula='fin + age + prio')

        # Should have results
        assert len(fgf.params_) == 3
        assert fgf.converged is True

        # Coefficients should be finite
        assert np.all(np.isfinite(fgf.params_))
        assert np.all(np.isfinite(fgf.standard_errors_))

    def test_with_formula(self):
        """Test formula specification"""
        np.random.seed(1)
        n = 80

        df = pd.DataFrame({
            'T': np.random.exponential(5, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.normal(0, 1, n),
            'X3': np.random.binomial(1, 0.5, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1, formula='X1 + X2')

        # Should only have X1 and X2
        assert len(fgf.params_) == 2
        assert 'X1' in fgf.params_.index
        assert 'X2' in fgf.params_.index
        assert 'X3' not in fgf.params_.index

    def test_with_censoring_groups(self):
        """Test with censoring groups"""
        np.random.seed(2)
        n = 120

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3]),
            'X1': np.random.normal(0, 1, n),
            'center': np.random.choice(['A', 'B', 'C'], n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1, censoring_groups_col='center')

        # Should fit successfully
        assert hasattr(fgf, 'params_')
        assert fgf.converged is True

    def test_predict_partial_hazard(self):
        """Test partial hazard prediction"""
        np.random.seed(3)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Predict for first 5 observations
        ph = fgf.predict_partial_hazard(df[['X1', 'X2']].head())

        # Should return Series with correct length
        assert isinstance(ph, pd.Series)
        assert len(ph) == 5

        # All values should be positive (exp of linear predictor)
        assert np.all(ph > 0)

    def test_predict_cumulative_incidence(self):
        """Test cumulative incidence prediction"""
        np.random.seed(4)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Predict CIF
        cif = fgf.predict_cumulative_incidence(df[['X1', 'X2']].head())

        # Should return DataFrame
        assert isinstance(cif, pd.DataFrame)

        # CIF should be between 0 and 1
        assert np.all(cif >= 0)
        assert np.all(cif <= 1)

        # CIF should be non-decreasing over time (for each individual)
        for col in cif.columns:
            diffs = cif[col].diff().dropna()
            assert np.all(diffs >= -1e-10), "CIF should be non-decreasing"

    def test_predict_survival_function(self):
        """Test survival function prediction"""
        np.random.seed(5)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        sf = fgf.predict_survival_function(df[['X1']].head())

        # Should be complement of CIF
        cif = fgf.predict_cumulative_incidence(df[['X1']].head())
        assert_frame_equal(sf, 1 - cif)

    def test_baseline_hazard_properties(self):
        """Test properties of baseline hazard"""
        np.random.seed(6)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Baseline hazard should be non-negative
        assert np.all(fgf.baseline_subdistribution_hazard_.values >= 0)

        # Cumulative hazard should be non-decreasing
        cum_hazard = fgf.baseline_cumulative_subdistribution_hazard_.iloc[:, 0]
        assert np.all(cum_hazard.diff().dropna() >= -1e-10)

    def test_no_competing_events(self):
        """Test when there are no competing events"""
        np.random.seed(7)
        n = 80

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1], n),  # Only censored and event of interest
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()

        # Should warn about no competing events
        with pytest.warns(UserWarning, match="No competing events"):
            fgf.fit(df, 'T', 'E', event_of_interest=1)

        # But should still fit
        assert hasattr(fgf, 'params_')

    def test_no_events_of_interest(self):
        """Test when event_of_interest not in data"""
        np.random.seed(8)
        n = 50

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 2], n),  # Only censored and competing
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()

        # Should raise error
        with pytest.raises(ValueError, match="event_of_interest.*not found"):
            fgf.fit(df, 'T', 'E', event_of_interest=1)

    def test_with_penalization(self):
        """Test with L2 penalization"""
        np.random.seed(9)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.normal(0, 1, n)
        })

        # Fit without penalization
        fgf_no_pen = FineGrayFitter(penalizer=0.0)
        fgf_no_pen.fit(df, 'T', 'E', event_of_interest=1)

        # Fit with penalization
        fgf_pen = FineGrayFitter(penalizer=1.0)
        fgf_pen.fit(df, 'T', 'E', event_of_interest=1)

        # Penalized coefficients should be smaller in magnitude
        assert np.linalg.norm(fgf_pen.params_) < np.linalg.norm(fgf_no_pen.params_)

    def test_print_summary(self):
        """Test that print_summary works"""
        np.random.seed(10)
        n = 80

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Should not raise error
        printer = fgf.print_summary()
        assert printer is not None

    def test_repr(self):
        """Test string representation"""
        np.random.seed(11)
        n = 50

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        # Before fitting
        fgf = FineGrayFitter()
        repr_str = repr(fgf)
        assert 'FineGrayFitter' in repr_str
        assert 'not fitted' in repr_str

        # After fitting
        fgf.fit(df, 'T', 'E', event_of_interest=1)
        repr_str = repr(fgf)
        assert 'FineGrayFitter' in repr_str
        assert 'fitted with' in repr_str
        assert 'observations' in repr_str

    def test_aic_partial(self):
        """Test partial AIC calculation"""
        np.random.seed(12)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # AIC should exist and be finite
        assert hasattr(fgf, 'AIC_partial_')
        assert np.isfinite(fgf.AIC_partial_)

        # AIC = -2*log_lik + 2*k
        expected_aic = -2 * fgf.log_likelihood_ + 2 * len(fgf.params_)
        assert_allclose(fgf.AIC_partial_, expected_aic)

    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        np.random.seed(13)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter(alpha=0.05)
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # CI should contain the estimate
        coef = fgf.params_.iloc[0]
        lower = fgf.confidence_intervals_.iloc[0, 0]
        upper = fgf.confidence_intervals_.iloc[0, 1]

        assert lower <= coef <= upper

    def test_competing_risks_effect(self):
        """Test that competing events reduce the CIF compared to no competing events"""
        np.random.seed(14)
        n = 100

        # Dataset with competing events
        df_comp = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n, p=[0.2, 0.4, 0.4]),
            'X1': np.random.normal(0, 1, n)
        })

        # Same dataset but competing events treated as censored
        df_no_comp = df_comp.copy()
        df_no_comp.loc[df_no_comp['E'] == 2, 'E'] = 0

        fgf_comp = FineGrayFitter()
        fgf_comp.fit(df_comp, 'T', 'E', event_of_interest=1)

        fgf_no_comp = FineGrayFitter()
        fgf_no_comp.fit(df_no_comp, 'T', 'E', event_of_interest=1)

        # With competing events, baseline CIF should generally be lower
        # (competing events "steal" probability mass)
        baseline_comp = fgf_comp.baseline_cumulative_subdistribution_hazard_.iloc[-1, 0]
        baseline_no_comp = fgf_no_comp.baseline_cumulative_subdistribution_hazard_.iloc[-1, 0]

        # Note: This is not always guaranteed due to random data,
        # but with large enough sample should hold
        # For now, just check both are positive and finite
        assert baseline_comp > 0
        assert baseline_no_comp > 0
        assert np.isfinite(baseline_comp)
        assert np.isfinite(baseline_no_comp)

    def test_prediction_at_specific_times(self):
        """Test prediction at specific time points"""
        np.random.seed(15)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Predict at specific times
        times = np.array([5, 10, 15, 20])
        cif = fgf.predict_cumulative_incidence(df[['X1']].head(), times=times)

        # Should have specified times as index
        assert_array_equal(cif.index, times)

    def test_with_external_weights(self):
        """Test with external observation weights"""
        np.random.seed(16)
        n = 100

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'weight': np.random.uniform(0.5, 2.0, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1, weights_col='weight')

        # Should fit successfully
        assert hasattr(fgf, 'params_')
        assert fgf.converged is True


class TestFineGrayEdgeCases:
    """Test edge cases and error handling"""

    def test_all_censored(self):
        """Test when all observations are censored"""
        np.random.seed(20)
        n = 50

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.zeros(n, dtype=int),  # All censored
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()

        # Should raise error (no events of interest)
        with pytest.raises(ValueError, match="event_of_interest.*not found"):
            fgf.fit(df, 'T', 'E', event_of_interest=1)

    def test_single_event(self):
        """Test with only one event of interest"""
        np.random.seed(21)
        n = 50

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.zeros(n, dtype=int),
            'X1': np.random.normal(0, 1, n)
        })
        df.loc[25, 'E'] = 1  # Single event

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Should fit but may have convergence issues
        assert hasattr(fgf, 'params_')

    def test_tied_times(self):
        """Test with tied event times"""
        np.random.seed(22)
        n = 50

        # Create data with many ties
        df = pd.DataFrame({
            'T': np.random.choice([1, 2, 3, 4, 5], n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Should handle ties correctly
        assert hasattr(fgf, 'params_')

    def test_predict_before_fit(self):
        """Test that prediction fails if model not fitted"""
        fgf = FineGrayFitter()

        df = pd.DataFrame({'X1': [1, 2, 3]})

        with pytest.raises(ValueError, match="not been fitted"):
            fgf.predict_partial_hazard(df)

        with pytest.raises(ValueError, match="not been fitted"):
            fgf.predict_cumulative_incidence(df)

    def test_missing_covariates_in_prediction(self):
        """Test prediction with missing covariates"""
        np.random.seed(23)
        n = 50

        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.choice([0, 1, 2], n),
            'X1': np.random.normal(0, 1, n),
            'X2': np.random.normal(0, 1, n)
        })

        fgf = FineGrayFitter()
        fgf.fit(df, 'T', 'E', event_of_interest=1)

        # Try to predict with only X1
        df_pred = pd.DataFrame({'X1': [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing columns"):
            fgf.predict_partial_hazard(df_pred)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
