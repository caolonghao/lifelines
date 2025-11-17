# -*- coding: utf-8 -*-
"""
Tests for CumulativeIncidenceFitter

These tests verify the implementation against expected behaviors and,
where possible, compare with R's cmprsk package results.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from lifelines import CumulativeIncidenceFitter
from lifelines.datasets import load_waltons


class TestCumulativeIncidenceFitter:
    """Tests for CumulativeIncidenceFitter"""

    def test_basic_fit_single_group(self):
        """Test basic fitting with a single group"""
        # Simple dataset
        times = np.array([1, 2, 2, 3, 4, 5])
        events = np.array([1, 1, 2, 0, 1, 2])  # 0=censored, 1=cause1, 2=cause2

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # Check that CIF is non-decreasing
        cif_values = cif.cumulative_incidence_.iloc[:, 0].values
        assert np.all(np.diff(cif_values) >= -1e-10), "CIF should be non-decreasing"

        # Check that CIF is between 0 and 1
        assert np.all(cif_values >= 0) and np.all(cif_values <= 1), "CIF should be in [0, 1]"

        # Check variance is non-negative
        var_values = cif.variance_.iloc[:, 0].values
        assert np.all(var_values >= 0), "Variance should be non-negative"

    def test_waltons_dataset(self):
        """Test with waltons dataset"""
        df = load_waltons()

        cif = CumulativeIncidenceFitter()
        cif.fit(df['T'], df['E'], event_of_interest=1)

        # Should have results
        assert len(cif.cumulative_incidence_) > 0
        assert len(cif.variance_) > 0

        # CIF should be reasonable
        final_cif = cif.cumulative_incidence_.iloc[-1, 0]
        assert 0 < final_cif < 1

    def test_multiple_groups(self):
        """Test with multiple groups"""
        np.random.seed(42)
        n = 100

        times = np.random.exponential(10, n)
        groups = np.random.choice(['A', 'B', 'C'], n)
        events = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1, groups=groups)

        # Should have estimates for each group
        assert cif.cumulative_incidence_.shape[1] == 3  # 3 groups

        # Should have test statistics
        assert cif.test_statistics_ is not None
        assert 'p_value' in cif.test_statistics_.columns
        assert 'statistic' in cif.test_statistics_.columns

    def test_grays_test_two_groups(self):
        """Test Gray's test with two groups"""
        np.random.seed(123)
        n_per_group = 50

        # Group A: higher risk of event 1
        times_a = np.random.exponential(10, n_per_group)
        events_a = np.random.choice([0, 1, 2], n_per_group, p=[0.2, 0.6, 0.2])

        # Group B: lower risk of event 1
        times_b = np.random.exponential(10, n_per_group)
        events_b = np.random.choice([0, 1, 2], n_per_group, p=[0.2, 0.3, 0.5])

        times = np.concatenate([times_a, times_b])
        events = np.concatenate([events_a, events_b])
        groups = np.array(['A'] * n_per_group + ['B'] * n_per_group)

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1, groups=groups)

        # Test should be performed
        assert cif.test_statistics_ is not None
        assert cif.test_statistics_['df'].iloc[0] == 1  # 2 groups - 1

        # p-value should be between 0 and 1
        p_val = cif.test_statistics_['p_value'].iloc[0]
        assert 0 <= p_val <= 1

    def test_no_events_of_interest(self):
        """Test when there are no events of interest"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([0, 2, 0, 2, 0])  # Only censored and event type 2

        cif = CumulativeIncidenceFitter()

        # Should raise error since event_of_interest=1 not in data
        with pytest.raises(ValueError, match="event_of_interest.*not found"):
            cif.fit(times, events, event_of_interest=1)

    def test_all_censored(self):
        """Test when all observations are censored"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([0, 0, 0, 0, 1])  # All censored except last

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # CIF should jump at last time only
        assert len(cif.cumulative_incidence_) == 1

    def test_variance_calculation(self):
        """Test that variance is calculated correctly"""
        np.random.seed(99)
        n = 100
        times = np.random.exponential(5, n)
        events = np.random.choice([0, 1, 2], n)

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # Variance should increase over time (generally)
        var_values = cif.variance_.iloc[:, 0].values

        # All variances should be non-negative
        assert np.all(var_values >= 0)

        # Standard errors should be finite
        se = np.sqrt(var_values)
        assert np.all(np.isfinite(se))

    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        times = np.array([1, 2, 3, 4, 5, 6])
        events = np.array([1, 1, 2, 1, 0, 2])

        cif = CumulativeIncidenceFitter(alpha=0.05)
        cif.fit(times, events, event_of_interest=1)

        # Should have confidence intervals
        assert hasattr(cif, 'confidence_interval_')
        assert cif.confidence_interval_.shape[1] == 2  # lower and upper

        # CI should contain the estimate
        cif_vals = cif.cumulative_incidence_.iloc[:, 0].values
        lower = cif.confidence_interval_.iloc[:, 0].values
        upper = cif.confidence_interval_.iloc[:, 1].values

        assert np.all(lower <= cif_vals)
        assert np.all(cif_vals <= upper)

    def test_weights(self):
        """Test with weights"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([1, 1, 2, 1, 0])
        weights = np.array([1, 2, 1, 1, 1])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1, weights=weights)

        # Should complete without error
        assert len(cif.cumulative_incidence_) > 0

    def test_competing_events_reduce_survival(self):
        """Test that competing events reduce overall survival"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([2, 2, 1, 0, 0])  # Competing events first

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # CIF for event 1 should be affected by early competing events
        # (implicitly tested through the algorithm)
        assert cif.cumulative_incidence_.iloc[-1, 0] < 1.0

    def test_plot_method(self):
        """Test that plot method works"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([1, 1, 2, 1, 0])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # Should not raise error
        ax = cif.plot()
        assert ax is not None

    def test_repr(self):
        """Test string representation"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([1, 1, 2, 1, 0])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        repr_str = repr(cif)
        assert 'CumulativeIncidenceFitter' in repr_str
        assert '5 observations' in repr_str

    def test_stratified_gray_test(self):
        """Test Gray's test with stratification"""
        np.random.seed(456)
        n = 200

        times = np.random.exponential(10, n)
        events = np.random.choice([0, 1, 2], n)
        groups = np.random.choice(['A', 'B'], n)
        strata = np.random.choice(['X', 'Y'], n)

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1, groups=groups, strata=strata)

        # Should complete without error
        assert cif.test_statistics_ is not None

    def test_comparison_with_aalen_johansen(self):
        """
        Compare CumulativeIncidenceFitter with AalenJohansenFitter.

        They should give similar results for the same data.
        """
        from lifelines import AalenJohansenFitter

        df = load_waltons()
        times = df['T'].values
        events = df['E'].values

        # CumulativeIncidenceFitter
        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # AalenJohansenFitter
        aj = AalenJohansenFitter(calculate_variance=True)
        aj.fit(times, events, event_of_interest=1)

        # Compare cumulative incidence at common time points
        common_times = np.intersect1d(
            cif.cumulative_incidence_.index,
            aj.cumulative_density_.index
        )

        if len(common_times) > 0:
            cif_values = cif.cumulative_incidence_.loc[common_times].iloc[:, 0].values
            aj_values = aj.cumulative_density_.loc[common_times].iloc[:, 0].values

            # Should be very close (allowing for small numerical differences)
            assert_allclose(cif_values, aj_values, rtol=0.01, atol=0.01)


class TestCumulativeIncidenceEdgeCases:
    """Test edge cases and error handling"""

    def test_single_event(self):
        """Test with only one event"""
        times = np.array([1, 2, 3, 4, 5])
        events = np.array([0, 0, 1, 0, 0])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        assert len(cif.cumulative_incidence_) == 1
        assert cif.cumulative_incidence_.iloc[0, 0] > 0

    def test_tied_times(self):
        """Test with tied event times"""
        times = np.array([1, 1, 1, 2, 2, 3])
        events = np.array([1, 1, 2, 1, 0, 2])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # Should handle ties correctly
        assert len(cif.cumulative_incidence_) > 0

    def test_large_dataset(self):
        """Test with a larger dataset"""
        np.random.seed(789)
        n = 1000

        times = np.random.exponential(10, n)
        events = np.random.choice([0, 1, 2, 3], n)

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # Should complete reasonably fast
        assert len(cif.cumulative_incidence_) > 0

    def test_numeric_stability(self):
        """Test numeric stability with very small/large values"""
        times = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        events = np.array([1, 1, 2, 1, 0])

        cif = CumulativeIncidenceFitter()
        cif.fit(times, events, event_of_interest=1)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(cif.cumulative_incidence_.values))
        assert np.all(np.isfinite(cif.variance_.values))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
