# -*- coding: utf-8 -*-
"""
Fine-Gray Proportional Subdistribution Hazards Model

Implementation of the Fine-Gray model for competing risks regression,
based on Fine & Gray (1999) JASA paper.

Reference implementation: R's cmprsk package (crr function)
"""

from __future__ import annotations

import time
import warnings
from typing import Optional, Union, List, Iterable

import numpy as np
import pandas as pd
from numpy import dot, exp, log
from numpy.linalg import inv, norm
from pandas import DataFrame, Series
from scipy.stats import chi2

from lifelines import KaplanMeierFitter
from lifelines.fitters import SemiParametricRegressionFitter
from lifelines import utils, exceptions


class FineGrayFitter(SemiParametricRegressionFitter):
    r"""
    Fine-Gray Proportional Subdistribution Hazards Model for Competing Risks.

    This class implements the Fine-Gray model which estimates the subdistribution
    hazard (cause-specific cumulative incidence) in the presence of competing risks.

    The subdistribution hazard at time t for the event of interest is modeled as:

    .. math::  \lambda_j(t|X) = \lambda_{j,0}(t) \exp(X \beta)

    where:
    - :math:`\lambda_{j,0}(t)` is the baseline subdistribution hazard
    - X are the covariates
    - :math:`\beta` are the regression coefficients

    The key feature is the use of inverse probability of censoring weighting (IPCW)
    to handle competing events, which allows subjects who experience competing events
    to remain in the risk set with appropriate weights.

    Parameters
    ----------
    alpha : float, optional (default=0.05)
        The alpha level for confidence intervals (1-alpha is the confidence level).

    penalizer : float or array, optional (default=0.0)
        Regularization parameter. Penalizes the size of the coefficients.
        Can be a single value or an array equal to the number of parameters.

    l1_ratio : float, optional (default=0.0)
        Specify elastic net penalty mixing ratio:
        l1_ratio=0 is ridge regression (L2 penalty)
        l1_ratio=1 is lasso regression (L1 penalty)
        0 < l1_ratio < 1 is elastic net

    Attributes
    ----------
    params_ : DataFrame
        The fitted coefficients and their standard errors.

    confidence_intervals_ : DataFrame
        The confidence intervals of the fitted parameters.

    summary : DataFrame
        A summary of the fit with coefficients, exp(coef), standard errors, z-values,
        and p-values.

    log_likelihood_ : float
        The partial log-likelihood of the fitted model.

    baseline_cumulative_subdistribution_hazard_ : DataFrame
        The baseline cumulative subdistribution hazard.

    baseline_subdistribution_hazard_ : DataFrame
        The baseline subdistribution hazard (jumps at event times).

    variance_matrix_ : DataFrame
        The variance-covariance matrix of the parameters.

    AIC_partial_ : float
        The AIC of the fitted model (using partial likelihood).

    converged : bool
        True if the Newton-Raphson algorithm converged.

    event_of_interest : int
        The event code for which the model was fitted.

    competing_event : int
        The event code(s) for competing events (all non-zero events except event_of_interest).

    Notes
    -----
    The Fine-Gray model differs from the standard Cox model in two key ways:

    1. **Modified risk set**: Subjects who experience competing events remain in the
       risk set after their event time, but with time-varying IPCW weights.

    2. **Different interpretation**: Coefficients represent the log subdistribution
       hazard ratio, which directly relates to cumulative incidence functions rather
       than cause-specific hazards.

    The IPCW weights are calculated as:
    - For subjects still at risk or censored: weight = exp(X*beta)
    - For subjects with competing events before current failure time:
      weight = exp(X*beta) * G(t_failure) / G(t_competing)
      where G(t) is the Kaplan-Meier estimate of the censoring distribution.

    References
    ----------
    Fine, J. P., & Gray, R. J. (1999). A Proportional Hazards Model for the
    Subdistribution of a Competing Risk. Journal of the American Statistical
    Association, 94(446), 496-509. DOI: 10.1080/01621459.1999.10474144

    Gray, R. J. (1988). A class of K-sample tests for comparing the cumulative
    incidence of a competing risk. Annals of Statistics, 16(3), 1141-1154.

    Examples
    --------
    >>> from lifelines import FineGrayFitter
    >>> from lifelines.datasets import load_rossi
    >>>
    >>> # Load data and create competing events
    >>> df = load_rossi()
    >>> # For demonstration: treat different event types
    >>> # In practice, your data should have proper event codes
    >>> # 0 = censored, 1 = event of interest, 2 = competing event
    >>>
    >>> fgf = FineGrayFitter()
    >>> fgf.fit(df, duration_col='week', event_col='arrest',
    ...         event_of_interest=1, formula='fin + age + prio')
    >>> fgf.print_summary()
    >>>
    >>> # Predict cumulative incidence for new subjects
    >>> fgf.predict_cumulative_incidence(df.head())

    See Also
    --------
    CoxPHFitter : Cox proportional hazards model (cause-specific hazards)
    CumulativeIncidenceFitter : Non-parametric cumulative incidence estimation
    """

    # Allowed residual types (subset of CoxPHFitter, some may not apply)
    _ALLOWED_RESIDUALS = {"schoenfeld", "score", "martingale", "deviance"}

    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: Union[float, np.ndarray] = 0.0,
        l1_ratio: float = 0.0,
    ):
        super(FineGrayFitter, self).__init__(alpha=alpha)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._censoring_groups_col = None

    def fit(
        self,
        df: DataFrame,
        duration_col: str,
        event_col: str,
        event_of_interest: int = 1,
        censoring_code: int = 0,
        censoring_groups_col: Optional[str] = None,
        show_progress: bool = False,
        step_size: float = 0.95,
        weights_col: Optional[str] = None,
        initial_point: Optional[np.ndarray] = None,
        formula: Optional[str] = None,
        fit_options: Optional[dict] = None,
    ) -> "FineGrayFitter":
        """
        Fit the Fine-Gray model to a dataset.

        Parameters
        ----------
        df : DataFrame
            The dataset containing duration, event, and covariates.

        duration_col : string
            The name of the column in the DataFrame that contains the durations.

        event_col : string
            The name of the column in the DataFrame that contains the event indicator.
            Event values should be coded as:
            - censoring_code (default 0) for censored observations
            - event_of_interest for the event of interest
            - any other positive integer for competing events

        event_of_interest : int, optional (default=1)
            The event code representing the event of interest for which to model
            the subdistribution hazard. All other non-zero event codes will be
            treated as competing events.

        censoring_code : int, optional (default=0)
            The event code representing censored observations.

        censoring_groups_col : string, optional (default=None)
            Column name for censoring groups. If provided, the censoring distribution
            will be estimated separately within each group. This should be used when
            you suspect the censoring mechanism differs across groups (e.g., different
            clinical centers, treatment arms, etc.).

        show_progress : bool, optional (default=False)
            Print convergence information during fitting.

        step_size : float, optional (default=0.95)
            Initial step size for Newton-Raphson optimization. Values between 0 and 1.
            Smaller values lead to more stable but slower convergence.

        weights_col : string, optional (default=None)
            Column name for observation weights. Note: These are NOT the IPCW weights,
            which are calculated internally. This is for external case weights.

        initial_point : array, optional (default=None)
            Initial coefficient values for optimization. If None, starts at zeros.

        formula : string, optional (default=None)
            An R-style formula for model specification. If provided, this is used to
            select covariates from df. Example: 'age + factor(sex) + treatment'
            If not provided, all columns except duration_col, event_col, weights_col,
            and censoring_groups_col will be used as covariates.

        fit_options : dict, optional (default=None)
            Additional options passed to the Newton-Raphson optimizer:
            - precision: convergence tolerance for parameter changes (default 1e-7)
            - max_steps: maximum number of iterations (default 500)
            - r_precision: relative convergence tolerance (default 1e-9)

        Returns
        -------
        self : FineGrayFitter
            The fitted model with parameter estimates.

        Examples
        --------
        >>> from lifelines import FineGrayFitter
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create example data with competing risks
        >>> np.random.seed(0)
        >>> n = 200
        >>> df = pd.DataFrame({
        ...     'T': np.random.exponential(10, n),
        ...     'E': np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3]),
        ...     'X1': np.random.normal(0, 1, n),
        ...     'X2': np.random.binomial(1, 0.5, n)
        ... })
        >>>
        >>> # Fit model for event type 1
        >>> fgf = FineGrayFitter()
        >>> fgf.fit(df, 'T', 'E', event_of_interest=1)
        >>> print(fgf.summary)

        >>> # With formula
        >>> fgf.fit(df, 'T', 'E', event_of_interest=1, formula='X1 + X2')

        >>> # With censoring groups
        >>> df['center'] = np.random.choice(['A', 'B', 'C'], n)
        >>> fgf.fit(df, 'T', 'E', event_of_interest=1,
        ...         censoring_groups_col='center')
        """

        # Store column names
        self.duration_col = duration_col
        self.event_col = event_col
        self.weights_col = weights_col
        self._censoring_groups_col = censoring_groups_col
        self.event_of_interest = event_of_interest
        self.censoring_code = censoring_code
        self._formula = formula

        # Set default fit options
        if fit_options is None:
            fit_options = {}
        fit_options.setdefault('precision', 1e-7)
        fit_options.setdefault('max_steps', 500)
        fit_options.setdefault('r_precision', 1e-9)

        # Preprocess data
        df = df.copy()

        # Handle formula if provided
        if formula is not None:
            # Use formulaic to parse formula and create design matrix
            import formulaic
            formula_obj = formulaic.Formula(formula)
            X = formula_obj.get_model_matrix(df)
            df = pd.concat([df[[duration_col, event_col]], X], axis=1)
            if weights_col:
                df[weights_col] = df[weights_col]
            if censoring_groups_col:
                df[censoring_groups_col] = df[censoring_groups_col]

        # Extract columns
        T = df[duration_col].copy()
        E = df[event_col].copy()

        # Get external weights (if provided)
        if weights_col:
            W_external = df[weights_col].values
        else:
            W_external = np.ones(len(df))

        # Get censoring groups
        if censoring_groups_col:
            censoring_groups = df[censoring_groups_col].copy()
            # Encode as integers
            unique_groups = sorted(censoring_groups.unique())
            group_map = {g: i+1 for i, g in enumerate(unique_groups)}
            censoring_groups_encoded = censoring_groups.map(group_map).values
            n_cen_groups = len(unique_groups)
            self._censoring_group_map = group_map
            self._censoring_groups_unique = unique_groups
        else:
            censoring_groups_encoded = np.ones(len(df), dtype=int)
            n_cen_groups = 1

        # Get covariate columns (exclude duration, event, weights, censoring groups)
        exclude_cols = [duration_col, event_col]
        if weights_col:
            exclude_cols.append(weights_col)
        if censoring_groups_col:
            exclude_cols.append(censoring_groups_col)
        X_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[X_cols].copy()

        # Check for valid data
        utils.check_nans_or_infs(T)
        utils.check_nans_or_infs(E)
        utils.check_nans_or_infs(X)
        utils.check_positivity(T)

        # Check event codes
        unique_events = sorted(E.unique())
        if event_of_interest not in unique_events:
            raise ValueError(
                f"event_of_interest={event_of_interest} not found in event column. "
                f"Available event codes: {unique_events}"
            )
        if censoring_code not in unique_events:
            warnings.warn(
                f"censoring_code={censoring_code} not found in data. "
                f"Available codes: {unique_events}",
                exceptions.StatisticalWarning
            )

        # Identify competing events
        competing_events = [e for e in unique_events
                           if e != censoring_code and e != event_of_interest]
        if len(competing_events) == 0:
            warnings.warn(
                "No competing events found in data. Consider using CoxPHFitter instead.",
                exceptions.StatisticalWarning
            )
        self.competing_events = competing_events

        # Recode events for internal use:
        # 0 = censored, 1 = event of interest, 2 = competing event
        E_internal = np.zeros_like(E.values)
        E_internal[E == event_of_interest] = 1
        for comp_event in competing_events:
            E_internal[E == comp_event] = 2

        # Sort data by time
        sort_idx = np.argsort(T.values)
        T_sorted = T.values[sort_idx]
        E_sorted = E_internal[sort_idx]
        X_sorted = X.values[sort_idx, :]
        W_external_sorted = W_external[sort_idx]
        censoring_groups_sorted = censoring_groups_encoded[sort_idx]

        # Store processed data
        self._T_sorted = T_sorted
        self._E_sorted = E_sorted
        self._X_sorted = X_sorted
        self._W_external_sorted = W_external_sorted
        self._censoring_groups_sorted = censoring_groups_sorted
        self._n_cen_groups = n_cen_groups
        self._X_columns = X.columns

        # Compute IPCW weights (censoring distribution)
        # This is G(t) = P(C > t) estimated by Kaplan-Meier
        self._censoring_weights_matrix = self._compute_censoring_weights(
            T_sorted, E_sorted, censoring_groups_sorted, n_cen_groups
        )

        # Normalize covariates for numerical stability
        self._norm_mean = X_sorted.mean(axis=0)
        self._norm_std = X_sorted.std(axis=0) + 1e-8  # avoid division by zero
        X_normalized = (X_sorted - self._norm_mean) / self._norm_std

        # Fit the model using Newton-Raphson
        n, d = X_normalized.shape

        if initial_point is None:
            beta = np.zeros(d)
        else:
            if len(initial_point) != d:
                raise ValueError(
                    f"initial_point must have length {d} (number of covariates)"
                )
            beta = initial_point / self._norm_std  # normalize initial point

        # Run Newton-Raphson optimization
        result = self._newton_raphson_fine_gray(
            X_normalized,
            T_sorted,
            E_sorted,
            W_external_sorted,
            censoring_groups_sorted,
            beta=beta,
            show_progress=show_progress,
            step_size=step_size,
            **fit_options
        )

        beta_final, log_lik, hessian, converged = result

        # Store convergence status
        self.converged = converged
        if not converged:
            warnings.warn(
                "Newton-Raphson failed to converge. Results may be unreliable. "
                "Try adjusting step_size, penalizer, or check for complete separation.",
                exceptions.ConvergenceWarning
            )

        # Rescale parameters back to original scale
        params = beta_final / self._norm_std

        # Compute variance matrix
        if hessian.size > 0:
            try:
                variance_matrix = -inv(hessian) / np.outer(self._norm_std, self._norm_std)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Hessian matrix is singular. Variance estimates may be unreliable.",
                    exceptions.StatisticalWarning
                )
                variance_matrix = np.full((d, d), np.nan)
        else:
            variance_matrix = np.full((d, d), np.nan)

        # Store results as DataFrames
        self.params_ = pd.Series(params, index=X.columns, name='coef')
        self.variance_matrix_ = pd.DataFrame(
            variance_matrix, index=X.columns, columns=X.columns
        )
        self.log_likelihood_ = log_lik
        self._hessian_ = hessian

        # Compute standard errors and confidence intervals
        self.standard_errors_ = pd.Series(
            np.sqrt(np.diag(variance_matrix)), index=X.columns, name='se(coef)'
        )

        z = self.params_ / self.standard_errors_
        self.p_values_ = pd.Series(
            2 * (1 - utils.ndtr(np.abs(z))), index=X.columns, name='p'
        )

        # Confidence intervals
        alpha_level = self.alpha
        z_critical = utils.inv_normal_cdf(1 - alpha_level / 2)

        lower = self.params_ - z_critical * self.standard_errors_
        upper = self.params_ + z_critical * self.standard_errors_

        self.confidence_intervals_ = pd.DataFrame({
            f'{100*(1-alpha_level)/2:.1f}%': lower,
            f'{100*(1+alpha_level)/2:.1f}%': upper
        })

        # Create summary DataFrame
        self.summary = pd.DataFrame({
            'coef': self.params_,
            'exp(coef)': np.exp(self.params_),
            'se(coef)': self.standard_errors_,
            'coef lower 95%': self.confidence_intervals_.iloc[:, 0],
            'coef upper 95%': self.confidence_intervals_.iloc[:, 1],
            'exp(coef) lower 95%': np.exp(self.confidence_intervals_.iloc[:, 0]),
            'exp(coef) upper 95%': np.exp(self.confidence_intervals_.iloc[:, 1]),
            'cmp to': 0,
            'z': z,
            'p': self.p_values_,
            '-log2(p)': -np.log2(self.p_values_)
        })

        # Compute baseline subdistribution hazard
        self._compute_baseline_subdistribution_hazard(
            X_normalized, T_sorted, E_sorted, W_external_sorted,
            censoring_groups_sorted, beta_final
        )

        # Store for predictions
        self._X_mean_for_predict = self._norm_mean
        self._X_std_for_predict = self._norm_std

        return self

    def _compute_censoring_weights(
        self,
        T: np.ndarray,
        E: np.ndarray,
        censoring_groups: np.ndarray,
        n_groups: int
    ) -> np.ndarray:
        """
        Compute IPCW weights using Kaplan-Meier estimation of censoring distribution.

        For each censoring group, estimate G(t) = P(C > t) where C is the censoring time.
        Returns a matrix of shape (n_groups, n_observations) where entry [g, i] contains
        the estimated probability that a subject in group g is uncensored at time T[i].

        This follows the approach in R's cmprsk::crr function.
        """
        n = len(T)
        weights_matrix = np.zeros((n_groups, n))

        for group_idx in range(1, n_groups + 1):
            # Get observations in this censoring group
            in_group = censoring_groups == group_idx

            if not in_group.any():
                # No observations in this group, set weights to 1
                weights_matrix[group_idx - 1, :] = 1.0
                continue

            T_group = T[in_group]
            E_group = E[in_group]

            # For censoring distribution, "event" is censoring (E==0)
            # and "censored" is any non-censored observation (E>0)
            censoring_event = (E_group == 0).astype(int)

            if censoring_event.sum() == 0:
                # No censoring events in this group
                weights_matrix[group_idx - 1, :] = 1.0
                continue

            # Fit Kaplan-Meier to censoring distribution
            kmf = KaplanMeierFitter()
            try:
                kmf.fit(T_group, censoring_event)

                # Evaluate at each time point (with small offset to get G(t-))
                # Use linear interpolation for times between observed points
                times_to_eval = T * (1 - 100 * np.finfo(float).eps)

                # Get survival probabilities at these times
                # KM survival function gives P(C > t)
                survival_at_times = np.interp(
                    times_to_eval,
                    kmf.survival_function_.index.values,
                    kmf.survival_function_.iloc[:, 0].values,
                    left=1.0,  # Before first time, survival = 1
                    right=0.0  # After last time, survival = 0
                )

                weights_matrix[group_idx - 1, :] = survival_at_times

            except Exception as e:
                warnings.warn(
                    f"Failed to compute censoring weights for group {group_idx}: {e}. "
                    "Using weights of 1.0",
                    exceptions.StatisticalWarning
                )
                weights_matrix[group_idx - 1, :] = 1.0

        return weights_matrix

    def _newton_raphson_fine_gray(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        W_external: np.ndarray,
        censoring_groups: np.ndarray,
        beta: np.ndarray,
        show_progress: bool = False,
        step_size: float = 0.95,
        precision: float = 1e-7,
        max_steps: int = 500,
        r_precision: float = 1e-9,
    ):
        """
        Newton-Raphson optimization for Fine-Gray model.

        This follows the algorithm described in Fine & Gray (1999) and implemented
        in R's cmprsk::crr function.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Normalized covariate matrix
        T : ndarray, shape (n,)
            Sorted event/censoring times
        E : ndarray, shape (n,)
            Event indicators (0=censored, 1=event of interest, 2=competing)
        W_external : ndarray, shape (n,)
            External case weights (not IPCW weights)
        censoring_groups : ndarray, shape (n,)
            Censoring group assignments
        beta : ndarray, shape (d,)
            Initial parameter values
        show_progress : bool
            Display convergence information
        step_size : float
            Initial step size for line search
        precision : float
            Convergence tolerance for parameter changes
        max_steps : int
            Maximum number of iterations
        r_precision : float
            Relative convergence tolerance for log-likelihood changes

        Returns
        -------
        beta : ndarray
            Optimized parameters
        log_lik : float
            Final log pseudo-likelihood
        hessian : ndarray
            Hessian matrix at convergence
        converged : bool
            Whether optimization converged
        """
        import autograd.numpy as anp
        from autograd import elementwise_grad

        n, d = X.shape

        # Penalization functions (for elastic net regularization)
        def soft_abs(x, a=1.0):
            return (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x)) / a

        def elastic_net_penalty(beta_val):
            if self.penalizer == 0:
                return 0.0
            penalty = self.penalizer * (
                self.l1_ratio * soft_abs(beta_val, 1.0).sum()
                + 0.5 * (1 - self.l1_ratio) * (beta_val ** 2).sum()
            )
            return n * penalty

        # Gradient of penalty
        d_elastic_net_penalty = elementwise_grad(elastic_net_penalty)
        # Second derivative of penalty
        dd_elastic_net_penalty = elementwise_grad(d_elastic_net_penalty)

        # Initialize
        converged = False
        log_lik = 0.0
        previous_log_lik = 0.0

        if show_progress:
            print(f"Newton-Raphson optimization for Fine-Gray model")
            print(f"{'Iter':<6} {'Log-Likelihood':<18} {'||Gradient||':<15} {'||Delta||':<15} {'Step Size':<12}")
            print("-" * 70)

        for iteration in range(max_steps):
            # Compute log-likelihood, gradient, and hessian at current beta
            log_lik, gradient, hessian = self._compute_log_likelihood_and_derivatives(
                X, T, E, W_external, censoring_groups, beta
            )

            # Add penalization
            if self.penalizer > 0:
                penalty = elastic_net_penalty(beta)
                log_lik -= penalty  # Negative because we're maximizing

                penalty_gradient = d_elastic_net_penalty(beta)
                gradient -= penalty_gradient

                penalty_hessian_diag = dd_elastic_net_penalty(beta)
                hessian -= np.diag(penalty_hessian_diag)

            # Check convergence (gradient-based)
            gradient_norm = norm(gradient)
            if iteration > 0:
                # Check both gradient convergence and relative likelihood change
                rel_ll_change = abs((log_lik - previous_log_lik) / (abs(log_lik) + 1e-10))

                if gradient_norm < precision or rel_ll_change < r_precision:
                    converged = True
                    if show_progress:
                        print(f"{iteration:<6} {log_lik:<18.6f} {gradient_norm:<15.8f} "
                              f"{'converged':<15} {'-':<12}")
                    break

            # Newton step: solve Hessian * delta = gradient
            try:
                delta = np.linalg.solve(-hessian, gradient)
            except np.linalg.LinAlgError:
                # Hessian is singular, try with regularization
                try:
                    delta = np.linalg.solve(-hessian + 1e-6 * np.eye(d), gradient)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        f"Hessian is singular at iteration {iteration}. "
                        "Cannot continue optimization.",
                        exceptions.ConvergenceWarning
                    )
                    break

            delta_norm = norm(delta)

            # Backtracking line search
            # Armijo condition: f(x + alpha*d) <= f(x) + c1 * alpha * grad'*d
            c1 = 1e-4
            max_backtracks = 20
            alpha = step_size
            beta_new = beta + alpha * delta

            # Evaluate at new point
            ll_new, _, _ = self._compute_log_likelihood_and_derivatives(
                X, T, E, W_external, censoring_groups, beta_new
            )
            if self.penalizer > 0:
                ll_new -= elastic_net_penalty(beta_new)

            armijo_threshold = log_lik + c1 * alpha * dot(gradient, delta)

            backtrack_count = 0
            while (np.isnan(ll_new) or ll_new < armijo_threshold) and backtrack_count < max_backtracks:
                alpha *= 0.5
                beta_new = beta + alpha * delta
                ll_new, _, _ = self._compute_log_likelihood_and_derivatives(
                    X, T, E, W_external, censoring_groups, beta_new
                )
                if self.penalizer > 0:
                    ll_new -= elastic_net_penalty(beta_new)
                armijo_threshold = log_lik + c1 * alpha * dot(gradient, delta)
                backtrack_count += 1

            if backtrack_count >= max_backtracks:
                warnings.warn(
                    f"Line search failed at iteration {iteration}. "
                    "Try reducing step_size or increasing penalizer.",
                    exceptions.ConvergenceWarning
                )
                break

            # Update
            beta = beta_new
            previous_log_lik = log_lik

            if show_progress:
                print(f"{iteration:<6} {log_lik:<18.6f} {gradient_norm:<15.8f} "
                      f"{delta_norm:<15.8f} {alpha:<12.6f}")

        if not converged and iteration >= max_steps - 1:
            warnings.warn(
                f"Newton-Raphson did not converge in {max_steps} iterations.",
                exceptions.ConvergenceWarning
            )

        # Final evaluation
        log_lik, _, hessian = self._compute_log_likelihood_and_derivatives(
            X, T, E, W_external, censoring_groups, beta
        )
        if self.penalizer > 0:
            hessian -= np.diag(dd_elastic_net_penalty(beta))

        return beta, log_lik, hessian, converged

    def _compute_log_likelihood_and_derivatives(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        W_external: np.ndarray,
        censoring_groups: np.ndarray,
        beta: np.ndarray,
    ):
        """
        Compute the log pseudo-likelihood, gradient, and hessian for Fine-Gray model.

        This implements the Fine-Gray partial likelihood with IPCW weighting,
        following the algorithm in R's cmprsk::crr function (crrfsv subroutine).

        Returns
        -------
        log_lik : float
            Log pseudo-likelihood (negative for minimization in original Fortran)
        gradient : ndarray, shape (d,)
            Gradient vector (score)
        hessian : ndarray, shape (d, d)
            Hessian matrix (observed information)
        """
        n, d = X.shape
        n_groups = self._n_cen_groups

        # Initialize
        log_lik = 0.0
        gradient = np.zeros(d)
        hessian = np.zeros((d, d))

        # Get unique failure times for event of interest (in descending order)
        failure_mask = E == 1
        unique_failure_times = np.unique(T[failure_mask])[::-1]  # Descending

        if len(unique_failure_times) == 0:
            # No events of interest
            return 0.0, gradient, hessian

        # Create index mapping for unique failure times
        failure_time_indices = {t: idx for idx, t in enumerate(unique_failure_times[::-1])}

        # Process each failure time (from latest to earliest, as in Fortran code)
        for failure_time in unique_failure_times:
            # Find all subjects who have event of interest at this time
            at_failure = (T == failure_time) & (E == 1)
            n_failures_at_time = at_failure.sum()

            if n_failures_at_time == 0:
                continue

            # Compute sum over failures at this time: sum(X_i * beta * W_i)
            X_at_failure = X[at_failure, :]
            W_at_failure = W_external[at_failure]

            linear_pred_failures = dot(X_at_failure, beta)
            weighted_lp_failures = (linear_pred_failures * W_at_failure).sum()

            # Contribution from failures: -sum(X*beta) (negative for pseudo-likelihood)
            log_lik -= weighted_lp_failures

            # Gradient contribution from failures: -sum(X_i * W_i)
            gradient -= dot(W_at_failure, X_at_failure)

            # Compute risk set contribution
            # Risk set includes:
            # 1. All subjects still at risk (T >= failure_time)
            # 2. Subjects with competing events BEFORE this failure time (with IPCW weights)

            risk_set_sum = 0.0
            risk_set_gradient = np.zeros(d)
            risk_set_hessian_component = np.zeros((d, d))

            for i in range(n):
                if T[i] < failure_time:
                    # Subject has event/censoring before this failure time
                    if E[i] <= 1:
                        # Censored or event of interest - not in risk set
                        continue
                    else:
                        # Competing event - remains in risk set with IPCW weight
                        # Weight = exp(X*beta) * G(t_failure) / G(t_competing)
                        group_idx = censoring_groups[i] - 1
                        # Find index for current failure time
                        failure_idx = np.searchsorted(T, failure_time)
                        G_at_failure = self._censoring_weights_matrix[group_idx, failure_idx]
                        G_at_competing = self._censoring_weights_matrix[group_idx, i]

                        if G_at_competing > 0:
                            ipcw_weight = G_at_failure / G_at_competing
                        else:
                            ipcw_weight = 0.0

                        linear_pred = dot(X[i, :], beta)
                        risk_contribution = exp(linear_pred) * W_external[i] * ipcw_weight

                else:
                    # Subject still at risk (T[i] >= failure_time)
                    linear_pred = dot(X[i, :], beta)
                    risk_contribution = exp(linear_pred) * W_external[i]

                # Accumulate risk set sums
                risk_set_sum += risk_contribution
                risk_set_gradient += risk_contribution * X[i, :]

                # For Hessian: outer product weighted by risk contribution
                # This computes sum_i w_i * exp(X_i*beta) * X_i * X_i^T
                risk_set_hessian_component += risk_contribution * np.outer(X[i, :], X[i, :])

            # Log-likelihood contribution from risk set
            if risk_set_sum > 0:
                log_lik += n_failures_at_time * log(risk_set_sum)

                # Gradient contribution: n_failures * (sum_risk X_i*w_i) / (sum_risk w_i)
                gradient += (n_failures_at_time / risk_set_sum) * risk_set_gradient

                # Hessian contribution (observed information)
                # H = n_failures * [ (sum w_i X_i X_i^T) / (sum w_i)
                #                    - (sum w_i X_i)(sum w_i X_i)^T / (sum w_i)^2 ]
                avg_X = risk_set_gradient / risk_set_sum
                hessian += (n_failures_at_time / risk_set_sum) * (
                    risk_set_hessian_component - np.outer(avg_X, avg_X) * risk_set_sum
                )

        return log_lik, gradient, hessian

    def _compute_baseline_subdistribution_hazard(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        W_external: np.ndarray,
        censoring_groups: np.ndarray,
        beta: np.ndarray,
    ):
        """
        Compute the baseline subdistribution hazard jumps.

        This follows the approach in R's cmprsk::crrfit function.

        The baseline subdistribution hazard jumps at each event time are:
        λ_0(t_j) = (# failures at t_j) / (sum of weighted risks at t_j)

        These are stored and used for predictions.
        """
        n, d = X.shape

        # Get unique failure times
        failure_mask = E == 1
        unique_failure_times = np.sort(np.unique(T[failure_mask]))

        if len(unique_failure_times) == 0:
            # No events of interest
            self.baseline_subdistribution_hazard_ = pd.DataFrame(
                {'baseline subdistribution hazard': []},
                index=pd.Index([], name='time')
            )
            self.baseline_cumulative_subdistribution_hazard_ = pd.DataFrame(
                {'baseline cumulative subdistribution hazard': []},
                index=pd.Index([], name='time')
            )
            return

        # Compute hazard jumps at each failure time
        hazard_jumps = []

        for failure_time in unique_failure_times:
            # Count failures at this time
            n_failures = ((T == failure_time) & (E == 1)).sum()

            # Compute risk set sum
            risk_set_sum = 0.0

            for i in range(n):
                if T[i] < failure_time:
                    if E[i] <= 1:
                        continue
                    else:
                        # Competing event with IPCW weight
                        group_idx = censoring_groups[i] - 1
                        failure_idx = np.searchsorted(T, failure_time)
                        G_at_failure = self._censoring_weights_matrix[group_idx, failure_idx]
                        G_at_competing = self._censoring_weights_matrix[group_idx, i]

                        if G_at_competing > 0:
                            ipcw_weight = G_at_failure / G_at_competing
                        else:
                            ipcw_weight = 0.0

                        linear_pred = dot(X[i, :], beta)
                        risk_set_sum += exp(linear_pred) * W_external[i] * ipcw_weight
                else:
                    # Still at risk
                    linear_pred = dot(X[i, :], beta)
                    risk_set_sum += exp(linear_pred) * W_external[i]

            # Hazard jump
            if risk_set_sum > 0:
                jump = n_failures / risk_set_sum
            else:
                jump = 0.0

            hazard_jumps.append(jump)

        # Store as DataFrames
        self.baseline_subdistribution_hazard_ = pd.DataFrame({
            'baseline subdistribution hazard': hazard_jumps
        }, index=unique_failure_times)
        self.baseline_subdistribution_hazard_.index.name = 'time'

        # Cumulative hazard
        self.baseline_cumulative_subdistribution_hazard_ = pd.DataFrame({
            'baseline cumulative subdistribution hazard':
                np.cumsum(hazard_jumps)
        }, index=unique_failure_times)
        self.baseline_cumulative_subdistribution_hazard_.index.name = 'time'

    def predict_partial_hazard(self, X: Union[DataFrame, Series]) -> Series:
        """
        Predict the partial hazard (exp(X*beta)) for new observations.

        Parameters
        ----------
        X : DataFrame or Series
            Covariates for prediction. Must have the same columns as training data.

        Returns
        -------
        partial_hazard : Series
            The predicted partial hazards exp(X*beta)
        """
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Ensure X is a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        # Check columns
        if not all(col in X.columns for col in self.params_.index):
            missing = [col for col in self.params_.index if col not in X.columns]
            raise ValueError(f"Missing columns in X: {missing}")

        X = X[self.params_.index]  # Reorder to match params

        # Normalize using training statistics
        X_normalized = (X - self._X_mean_for_predict) / self._X_std_for_predict

        # Compute linear predictor
        linear_pred = dot(X_normalized, self.params_.values / self._X_std_for_predict)

        # Return exp(X*beta)
        return pd.Series(np.exp(linear_pred), index=X.index, name='partial_hazard')

    def predict_cumulative_incidence(
        self,
        X: Union[DataFrame, Series],
        times: Optional[Iterable] = None
    ) -> DataFrame:
        """
        Predict the cumulative incidence function for new observations.

        The cumulative incidence (subdistribution function) is:
        F(t|X) = 1 - exp(-Λ_0(t) * exp(X*beta))

        where Λ_0(t) is the baseline cumulative subdistribution hazard.

        Parameters
        ----------
        X : DataFrame or Series
            Covariates for prediction.

        times : array-like, optional
            Times at which to evaluate the cumulative incidence.
            If None, uses the unique event times from the training data.

        Returns
        -------
        cumulative_incidence : DataFrame
            Predicted cumulative incidence function.
            Rows are times, columns are observations.
        """
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Ensure X is a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        # Get partial hazards
        partial_hazards = self.predict_partial_hazard(X)

        # Get baseline cumulative hazard
        baseline_cum_hazard = self.baseline_cumulative_subdistribution_hazard_.iloc[:, 0]

        # Determine times
        if times is None:
            times = baseline_cum_hazard.index.values
        else:
            times = np.atleast_1d(times)

        # Interpolate baseline cumulative hazard at requested times
        baseline_at_times = np.interp(
            times,
            baseline_cum_hazard.index.values,
            baseline_cum_hazard.values,
            left=0.0,
            right=baseline_cum_hazard.values[-1]
        )

        # Compute cumulative incidence for each observation
        # CIF(t|X) = 1 - exp(-Λ_0(t) * exp(X*beta))
        cum_inc = np.outer(baseline_at_times, partial_hazards.values)
        cum_inc = 1 - np.exp(-cum_inc)

        # Return as DataFrame
        return pd.DataFrame(
            cum_inc,
            index=times,
            columns=X.index
        )

    def predict_survival_function(
        self,
        X: Union[DataFrame, Series],
        times: Optional[Iterable] = None
    ) -> DataFrame:
        """
        Predict the survival function (1 - cumulative incidence) for new observations.

        Note: This is the probability of NOT experiencing the event of interest by time t,
        but subjects may have experienced competing events.

        Parameters
        ----------
        X : DataFrame or Series
            Covariates for prediction.

        times : array-like, optional
            Times at which to evaluate.

        Returns
        -------
        survival_function : DataFrame
            Predicted survival function (1 - CIF).
        """
        cum_inc = self.predict_cumulative_incidence(X, times)
        return 1 - cum_inc

    def print_summary(self, decimals: int = 2, **kwargs):
        """
        Print a summary of the fitted model.

        Parameters
        ----------
        decimals : int, optional (default=2)
            Number of decimal places to display.

        **kwargs
            Additional arguments passed to utils.Printer
        """
        if not hasattr(self, 'params_'):
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        from lifelines.utils.printer import Printer

        printer = Printer(self, decimals=decimals, **kwargs)

        # Header
        printer.print(f"<lifelines.FineGrayFitter: fitted with {len(self._T_sorted)} total observations, "
                     f"{(self._E_sorted == 1).sum()} events of interest, "
                     f"{(self._E_sorted == 2).sum()} competing events>")
        printer.print("")

        # Model fit statistics
        printer.print(f"{'Event of interest code':<30} = {self.event_of_interest}")
        printer.print(f"{'Competing event code(s)':<30} = {self.competing_events}")
        printer.print(f"{'Partial log-likelihood':<30} = {self.log_likelihood_:.{decimals}f}")
        printer.print(f"{'Time fit was run':<30} = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        printer.print("")
        printer.print("---")

        # Coefficients table
        justify = utils.string_justify(18)

        headers = [
            "",
            justify("coef"),
            justify("exp(coef)"),
            justify("se(coef)"),
            justify("coef lower 95%"),
            justify("coef upper 95%"),
            justify("exp(coef) lower 95%"),
            justify("exp(coef) upper 95%"),
            justify("cmp to"),
            justify("z"),
            justify("p"),
            justify("-log2(p)"),
        ]
        printer.print(" ".join(headers))

        for param in self.params_.index:
            fmt = utils.format_floats(decimals)
            row = [
                param.ljust(20),
                fmt(self.summary.loc[param, 'coef']),
                fmt(self.summary.loc[param, 'exp(coef)']),
                fmt(self.summary.loc[param, 'se(coef)']),
                fmt(self.summary.loc[param, 'coef lower 95%']),
                fmt(self.summary.loc[param, 'coef upper 95%']),
                fmt(self.summary.loc[param, 'exp(coef) lower 95%']),
                fmt(self.summary.loc[param, 'exp(coef) upper 95%']),
                fmt(self.summary.loc[param, 'cmp to']),
                fmt(self.summary.loc[param, 'z']),
                fmt(self.summary.loc[param, 'p']),
                fmt(self.summary.loc[param, '-log2(p)'])
            ]
            printer.print(" ".join(row))

        printer.print("---")

        # Concordance and AIC
        printer.print(f"Partial AIC = {self.AIC_partial_:.{decimals}f}")

        if not self.converged:
            printer.print("")
            printer.print("Warning: Optimization did not converge. Results may be unreliable.")

        return printer

    def __repr__(self) -> str:
        """String representation of the fitter."""
        if hasattr(self, 'params_'):
            return (f"<lifelines.FineGrayFitter: fitted with {len(self._T_sorted)} observations, "
                   f"{(self._E_sorted == 1).sum()} events of interest>")
        else:
            return "<lifelines.FineGrayFitter: not fitted>"
