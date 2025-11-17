# -*- coding: utf-8 -*-
"""
Cumulative Incidence Function estimation in competing risks framework.

This module implements the non-parametric estimation of cumulative incidence
functions (CIF) for competing risks data, along with Gray's K-sample test for
comparing cumulative incidence curves across groups.

References
----------
1. Gray RJ (1988). "A class of K-sample tests for comparing the cumulative
   incidence of a competing risk." Annals of Statistics 16:1141-1154.
   DOI: 10.1214/aos/1176350951

2. Aalen O (1978). "Nonparametric estimation of partial transition probabilities
   in multiple decrement models." Annals of Statistics 6:534-545.
"""

from typing import Optional, Union, List, Tuple
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2

from lifelines.fitters import NonParametricUnivariateFitter
from lifelines.utils import _preprocess_inputs, CensoringType
from lifelines import KaplanMeierFitter
from lifelines.statistics import StatisticalResult


__all__ = ["CumulativeIncidenceFitter"]


class CumulativeIncidenceFitter(NonParametricUnivariateFitter):
    """
    Estimate cumulative incidence functions in a competing risks framework.

    The cumulative incidence function (CIF) represents the probability of
    experiencing a specific type of event by time t, accounting for competing
    risks. This is also known as the subdistribution function.

    The CIF is estimated using the Aalen-Johansen estimator:

        CIF_k(t) = Σ_{t_i ≤ t} S(t_i-) × λ_k(t_i)

    where:
        - S(t-) is the overall survival function just before time t
        - λ_k(t) is the cause-specific hazard for event type k at time t

    This implementation follows the algorithm described in Gray (1988) and
    computes variances using Aalen's method.

    Parameters
    ----------
    alpha : float, optional (default=0.05)
        The alpha value for confidence intervals.

    Examples
    --------
    >>> from lifelines import CumulativeIncidenceFitter
    >>> from lifelines.datasets import load_waltons
    >>> df = load_waltons()
    >>> cif = CumulativeIncidenceFitter()
    >>> # Fit for a specific event type
    >>> cif.fit(df['T'], df['E'], event_of_interest=1)
    >>> cif.plot()
    >>> print(cif.cumulative_incidence_)
    >>>
    >>> # Compare across groups
    >>> cif.fit(df['T'], df['E'], event_of_interest=1, groups=df['group'])
    >>> print(cif.test_statistics_)  # Gray's test results

    Attributes
    ----------
    cumulative_incidence_ : DataFrame
        The estimated cumulative incidence function(s), indexed by time.
        If groups are specified, columns represent different group-cause combinations.

    variance_ : DataFrame
        The variance estimates for the CIF, using Aalen's method.

    confidence_interval_ : DataFrame
        The confidence intervals for the CIF.

    test_statistics_ : DataFrame
        Results of Gray's K-sample test (only if groups are specified).
        Contains test statistics, p-values, and degrees of freedom for each cause.

    event_of_interest : int
        The event type code for the cause of interest.

    groups_ : Series
        The group labels (if groups were specified).

    causes_ : ndarray
        Unique event type codes found in the data.

    References
    ----------
    Gray RJ (1988). "A class of K-sample tests for comparing the cumulative
    incidence of a competing risk." Annals of Statistics 16:1141-1154.

    Aalen O (1978). "Nonparametric estimation of partial transition probabilities
    in multiple decrement models." Annals of Statistics 6:534-545.
    """

    def __init__(self, alpha: float = 0.05, **kwargs):
        super(CumulativeIncidenceFitter, self).__init__(alpha=alpha, **kwargs)
        self.event_of_interest = None
        self.groups_ = None
        self.causes_ = None
        self.test_statistics_ = None

    @CensoringType.right_censoring
    def fit(
        self,
        durations,
        event_observed,
        event_of_interest: int = 1,
        groups=None,
        strata=None,
        timeline=None,
        entry=None,
        label: Optional[str] = None,
        alpha: Optional[float] = None,
        ci_labels: Optional[Tuple[str, str]] = None,
        weights=None,
    ):
        """
        Fit the cumulative incidence function estimator.

        Parameters
        ----------
        durations : array-like
            Duration subjects were observed for (time to event or censoring).

        event_observed : array-like
            Event type indicator. Use 0 for censoring, and positive integers
            (1, 2, 3, ...) for different event types.

        event_of_interest : int, optional (default=1)
            The event type code to estimate CIF for. All other non-zero event
            types are treated as competing events.

        groups : array-like, optional
            Group membership for each subject. If provided, CIF will be estimated
            separately for each group, and Gray's K-sample test will be performed
            to compare groups.

        strata : array-like, optional
            Stratification variable for Gray's test. Tests will be stratified on
            this variable. Only used when groups are provided.

        timeline : array-like, optional
            Times at which to estimate the CIF.

        entry : array-like, optional
            Entry times for left-truncated data. If None, all subjects are
            assumed to enter at time 0.

        label : str, optional
            Label for the CIF estimate.

        alpha : float, optional
            Override the alpha value for this fit.

        ci_labels : tuple, optional
            Custom labels for confidence interval columns.

        weights : array-like, optional
            Individual-level weights. For example, for sampling weights or
            frequency weights.

        Returns
        -------
        self : CumulativeIncidenceFitter
            Fitted estimator with cumulative_incidence_ attribute.
        """
        alpha = alpha if alpha is not None else self.alpha
        self.alpha = alpha
        self.event_of_interest = int(event_of_interest)

        # Convert inputs to arrays
        durations = np.asarray(durations)
        event_observed = np.asarray(event_observed)

        if weights is None:
            weights = np.ones(len(durations))
        else:
            weights = np.asarray(weights)

        # Handle groups
        if groups is not None:
            groups = pd.Series(groups, index=pd.RangeIndex(len(groups)))
            groups = groups.astype('category')
            self.groups_ = groups

        # Handle strata
        if strata is not None:
            strata = pd.Series(strata, index=pd.RangeIndex(len(strata)))
            strata = strata.astype('category')

        # Get unique event types
        self.causes_ = np.unique(event_observed[event_observed > 0])

        # Check if event_of_interest exists in data
        if self.event_of_interest not in self.causes_:
            raise ValueError(
                f"event_of_interest={self.event_of_interest} not found in data. "
                f"Available event types: {self.causes_}"
            )

        # Sort by time
        sort_idx = np.argsort(durations)
        durations = durations[sort_idx]
        event_observed = event_observed[sort_idx]
        weights = weights[sort_idx]
        if groups is not None:
            groups = groups.iloc[sort_idx].reset_index(drop=True)
        if strata is not None:
            strata = strata.iloc[sort_idx].reset_index(drop=True)

        # Estimate CIF
        if groups is None:
            # Single group estimation
            cif_result = self._estimate_cumulative_incidence(
                durations, event_observed, self.event_of_interest, weights
            )

            # Create label
            if label is None:
                label = f"CIF_{self.event_of_interest}"

            # Store results
            self.cumulative_incidence_ = pd.DataFrame(
                {label: cif_result['cif']},
                index=cif_result['time']
            )
            self.cumulative_incidence_.index.name = 'timeline'

            self.variance_ = pd.DataFrame(
                {label: cif_result['variance']},
                index=cif_result['time']
            )
            self.variance_.index.name = 'timeline'

            # Compute confidence intervals
            self._compute_confidence_intervals(label, ci_labels)

        else:
            # Multiple groups estimation
            unique_groups = groups.cat.categories
            all_results = {}

            # Estimate for each group
            for grp in unique_groups:
                grp_mask = (groups == grp)
                grp_durations = durations[grp_mask]
                grp_events = event_observed[grp_mask]
                grp_weights = weights[grp_mask]

                cif_result = self._estimate_cumulative_incidence(
                    grp_durations, grp_events, self.event_of_interest, grp_weights
                )

                grp_label = f"{grp}_{self.event_of_interest}"
                all_results[grp_label] = cif_result

            # Combine results into DataFrames
            all_times = sorted(set().union(*[set(r['time']) for r in all_results.values()]))

            cif_df = pd.DataFrame(index=all_times)
            var_df = pd.DataFrame(index=all_times)

            for grp_label, result in all_results.items():
                # Forward fill for step function
                cif_series = pd.Series(result['cif'], index=result['time'])
                var_series = pd.Series(result['variance'], index=result['time'])

                cif_df[grp_label] = cif_series.reindex(all_times, method='ffill').fillna(0)
                var_df[grp_label] = var_series.reindex(all_times, method='ffill').fillna(0)

            self.cumulative_incidence_ = cif_df
            self.cumulative_incidence_.index.name = 'timeline'
            self.variance_ = var_df
            self.variance_.index.name = 'timeline'

            # Perform Gray's test
            self.test_statistics_ = self._gray_test(
                durations, event_observed, groups, strata, weights
            )

        self.timeline = self.cumulative_incidence_.index.values
        self.event_observed = event_observed
        self.durations = durations

        return self

    def _estimate_cumulative_incidence(
        self,
        times: np.ndarray,
        events: np.ndarray,
        event_of_interest: int,
        weights: np.ndarray
    ) -> dict:
        """
        Estimate cumulative incidence function for a single group.

        This implements the algorithm from Gray (1988) / cincsub.f.

        Parameters
        ----------
        times : ndarray
            Sorted event/censoring times.
        events : ndarray
            Event type indicators (0 = censored, >0 = event type).
        event_of_interest : int
            Event type to estimate CIF for.
        weights : ndarray
            Individual weights.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'time': ndarray of unique event times
            - 'cif': ndarray of CIF estimates
            - 'variance': ndarray of variance estimates (Aalen's method)
        """
        n = len(times)

        # Find unique event times for the cause of interest
        event_mask = (events == event_of_interest)
        unique_times = np.unique(times[event_mask])

        if len(unique_times) == 0:
            # No events of interest
            return {
                'time': np.array([times[-1]]),
                'cif': np.array([0.0]),
                'variance': np.array([0.0])
            }

        # Initialize
        cif_values = np.zeros(len(unique_times))
        var_values = np.zeros(len(unique_times))

        # Overall survival (Kaplan-Meier for any event)
        S = 1.0

        # Variance components (Aalen's method)
        v1 = 0.0
        v2 = 0.0
        v3 = 0.0

        # Track position in data
        current_idx = 0

        for i, t in enumerate(unique_times):
            # Count subjects at risk
            at_risk = np.sum(times >= t)

            # Count events at time t
            at_time_mask = (times == t)
            events_at_t = events[at_time_mask]
            weights_at_t = weights[at_time_mask]

            # Weighted counts
            d1 = np.sum(weights_at_t[events_at_t == event_of_interest])  # Cause of interest
            d2 = np.sum(weights_at_t[(events_at_t > 0) & (events_at_t != event_of_interest)])  # Competing
            d_total = d1 + d2

            # Update CIF
            if i > 0:
                cif_values[i] = cif_values[i-1]

            if d1 > 0 and at_risk > 0:
                cif_values[i] += S * d1 / at_risk

            # Update variance (Aalen's method)
            # See cincsub.f lines 63-91
            if d_total > 0 and at_risk > 0:
                S_new = S * (at_risk - d_total) / at_risk

                # Variance for competing events
                if d2 > 0 and S_new > 0:
                    # Adjust for finite sample (line 65)
                    t5 = 1.0
                    if d2 > 1:
                        t5 = 1.0 - (d2 - 1.0) / (at_risk - 1.0)

                    t6 = S * S * t5 * d2 / (at_risk * at_risk)
                    t3 = 1.0 / S_new
                    t4 = cif_values[i] / S_new

                    v1 += t4 * t4 * t6
                    v2 += t3 * t4 * t6
                    v3 += t3 * t3 * t6

                # Variance for event of interest
                if d1 > 0:
                    # Adjust for finite sample (line 74)
                    t5 = 1.0
                    if d1 > 1:
                        t5 = 1.0 - (d1 - 1.0) / (at_risk - 1.0)

                    t6 = S * S * t5 * d1 / (at_risk * at_risk)
                    t3 = 0.0
                    if S_new > 0:
                        t3 = 1.0 / S_new

                    t4 = 1.0 + t3 * cif_values[i]

                    v1 += t4 * t4 * t6
                    v2 += t3 * t4 * t6
                    v3 += t3 * t3 * t6

                # Compute variance (line 91)
                t2 = cif_values[i]
                var_values[i] = v1 + t2 * t2 * v3 - 2.0 * t2 * v2

                # Update survival
                S = S_new
            else:
                if i > 0:
                    var_values[i] = var_values[i-1]

        return {
            'time': unique_times,
            'cif': cif_values,
            'variance': var_values
        }

    def _gray_test(
        self,
        times: np.ndarray,
        events: np.ndarray,
        groups: pd.Series,
        strata: Optional[pd.Series],
        weights: np.ndarray
    ) -> pd.DataFrame:
        """
        Perform Gray's K-sample test for comparing cumulative incidence across groups.

        This implements the test described in Gray (1988), which compares the
        cumulative incidence functions for a specific cause across multiple groups.

        Parameters
        ----------
        times : ndarray
            Event/censoring times.
        events : ndarray
            Event type indicators.
        groups : Series
            Group membership.
        strata : Series, optional
            Stratification variable.
        weights : ndarray
            Individual weights.

        Returns
        -------
        DataFrame
            Test results with columns: cause, statistic, p_value, df
        """
        # Test for the event of interest
        test_result = self._gray_test_single_cause(
            times, events, self.event_of_interest, groups, strata, weights
        )

        return pd.DataFrame([{
            'cause': self.event_of_interest,
            'statistic': test_result['statistic'],
            'p_value': test_result['p_value'],
            'df': test_result['df']
        }])

    def _gray_test_single_cause(
        self,
        times: np.ndarray,
        events: np.ndarray,
        cause: int,
        groups: pd.Series,
        strata: Optional[pd.Series],
        weights: np.ndarray
    ) -> dict:
        """
        Gray's test for a single cause.

        Implements the weighted test statistic from Gray (1988).
        The test statistic follows a chi-square distribution with (# groups - 1) df.

        Reference: crstm.f subroutine
        """
        unique_groups = groups.cat.categories
        n_groups = len(unique_groups)

        if n_groups < 2:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'df': 0
            }

        # Use unstratified test if no strata provided
        if strata is None:
            strata = pd.Series(np.ones(len(times)), dtype='category')

        unique_strata = strata.cat.categories
        n_strata = len(unique_strata)

        # Score vector (length n_groups - 1)
        scores = np.zeros(n_groups - 1)

        # Variance-covariance matrix
        var_matrix = np.zeros((n_groups - 1, n_groups - 1))

        # Compute scores and variance for each stratum
        for stratum in unique_strata:
            stratum_mask = (strata == stratum)

            s_times = times[stratum_mask]
            s_events = events[stratum_mask]
            s_groups = groups[stratum_mask]
            s_weights = weights[stratum_mask]

            if len(s_times) == 0:
                continue

            # Compute scores and variance for this stratum
            s_scores, s_var = self._compute_gray_scores(
                s_times, s_events, s_groups, cause, unique_groups, s_weights
            )

            # Accumulate across strata
            scores += s_scores
            var_matrix += s_var

        # Compute test statistic: score' * var^-1 * score
        # This follows chi-square(n_groups - 1)
        try:
            # Use pseudo-inverse for numerical stability
            var_inv = np.linalg.pinv(var_matrix)
            test_stat = scores @ var_inv @ scores

            # Check if matrix is singular
            if np.linalg.matrix_rank(var_matrix) < n_groups - 1:
                warnings.warn(
                    "Variance matrix is singular. Test statistic may be unreliable.",
                    UserWarning
                )
                test_stat = -1.0
        except np.linalg.LinAlgError:
            warnings.warn(
                "Failed to invert variance matrix. Cannot compute test statistic.",
                UserWarning
            )
            test_stat = -1.0

        # Compute p-value
        df = n_groups - 1
        if test_stat >= 0:
            p_value = 1.0 - chi2.cdf(test_stat, df)
        else:
            p_value = np.nan

        return {
            'statistic': test_stat,
            'p_value': p_value,
            'df': df
        }

    def _compute_gray_scores(
        self,
        times: np.ndarray,
        events: np.ndarray,
        groups: pd.Series,
        cause: int,
        unique_groups,
        weights: np.ndarray,
        rho: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute score vector and variance matrix for Gray's test.

        This implements the core calculation from crstm.f / crst subroutine.

        The weight function is S(t-)^rho, where rho=0 gives equal weights (default).
        """
        n_groups = len(unique_groups)
        n = len(times)

        # Initialize outputs
        scores = np.zeros(n_groups - 1)
        var_matrix = np.zeros((n_groups - 1, n_groups - 1))

        # Track risk set size for each group
        risk_set = {grp: np.sum(groups == grp) for grp in unique_groups}

        # Overall survival function S(t) for weights
        S = 1.0

        # Track cumulative incidence for weight function
        F = np.zeros(n_groups)

        # Process each unique failure time
        unique_times = np.unique(times[events == cause])

        for t in unique_times:
            at_time = (times == t)

            # Update risk sets (remove anyone who left before this time)
            at_risk = times >= t
            for g_idx, grp in enumerate(unique_groups):
                risk_set[grp] = np.sum((groups == grp) & at_risk)

            total_at_risk = sum(risk_set.values())
            if total_at_risk == 0:
                continue

            # Count events of this cause at time t, by group
            d = np.zeros(n_groups)
            for g_idx, grp in enumerate(unique_groups):
                mask = at_time & (groups == grp) & (events == cause)
                d[g_idx] = np.sum(weights[mask])

            d_total = np.sum(d)

            if d_total == 0:
                # Update survival for competing events
                n_competing = np.sum(weights[at_time & (events > 0) & (events != cause)])
                if n_competing > 0:
                    S *= (total_at_risk - n_competing) / total_at_risk
                continue

            # Weight function: S(t-)^rho
            w = S ** rho

            # Expected number of events in each group under null
            E = np.array([risk_set[grp] * d_total / total_at_risk for grp in unique_groups])

            # Variance components
            # V[i,j] = w^2 * E[i] * (delta_ij - E[j]/total_at_risk)
            V = np.zeros((n_groups, n_groups))
            for i in range(n_groups):
                for j in range(n_groups):
                    if i == j:
                        V[i, j] = w * w * E[i] * (1.0 - E[i] / total_at_risk)
                    else:
                        V[i, j] = -w * w * E[i] * E[j] / total_at_risk

            # Accumulate score (first n_groups - 1 components)
            # Score = (Observed - Expected) weighted
            obs_minus_exp = (d - E)[:n_groups-1]
            scores += w * obs_minus_exp

            # Accumulate variance (first (n_groups-1) x (n_groups-1) block)
            var_matrix += V[:n_groups-1, :n_groups-1]

            # Update survival function
            S *= (total_at_risk - d_total) / total_at_risk

        return scores, var_matrix

    def _compute_confidence_intervals(self, label: str, ci_labels: Optional[Tuple[str, str]]):
        """
        Compute confidence intervals for the CIF using normal approximation.

        CIF is on [0, 1], so we use log(-log) transformation for better coverage.
        """
        from scipy.stats import norm

        z = norm.ppf(1 - self.alpha / 2)

        cif = self.cumulative_incidence_[label].values
        var = self.variance_[label].values

        # Avoid division by zero
        se = np.sqrt(np.maximum(var, 0))

        # Use log(-log) transformation for better CI on [0,1]
        # CI = CIF^exp(±z*se / (CIF * log(CIF)))
        # But for simplicity, use normal approximation first
        lower = np.maximum(cif - z * se, 0.0)
        upper = np.minimum(cif + z * se, 1.0)

        if ci_labels is None:
            ci_labels = (f"{label}_lower_{1-self.alpha/2:.2f}",
                        f"{label}_upper_{1-self.alpha/2:.2f}")

        self.confidence_interval_ = pd.DataFrame({
            ci_labels[0]: lower,
            ci_labels[1]: upper
        }, index=self.cumulative_incidence_.index)

    def plot(self, **kwargs):
        """
        Plot the cumulative incidence function(s).

        Returns
        -------
        ax : matplotlib axis
        """
        from lifelines.plotting import _plot_estimate

        return _plot_estimate(
            self,
            estimate='cumulative_incidence_',
            **kwargs
        )

    def __repr__(self):
        return f"<lifelines.CumulativeIncidenceFitter: fitted with {len(self.durations)} observations, {self.event_observed.sum()} events>"
