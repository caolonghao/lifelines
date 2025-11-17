# CLAUDE.md - Lifelines Codebase Guide for AI Assistants

**Version:** 0.30.0
**Last Updated:** 2025-11-17

## Project Overview

**Lifelines** is a pure Python implementation of survival analysis tools, originally developed for actuarial and medical research but widely applicable to any domain measuring time-to-event data. The library provides non-parametric, semi-parametric, and parametric models for survival analysis.

- **License:** MIT
- **Language:** Python (>=3.9)
- **Repository:** https://github.com/CamDavidsonPilon/lifelines
- **Documentation:** http://lifelines.readthedocs.org/
- **Maintainer:** Cameron Davidson-Pilon

## Codebase Structure

```
/home/user/lifelines/
├── lifelines/                    # Main package (~18K lines)
│   ├── __init__.py               # Package exports and version
│   ├── fitters/                  # Core survival models (~3.6K lines)
│   │   ├── __init__.py           # Base classes and fitter hierarchy
│   │   ├── kaplan_meier_fitter.py
│   │   ├── nelson_aalen_fitter.py
│   │   ├── coxph_fitter.py       # Cox PH model (largest: 3.2K lines)
│   │   ├── cox_time_varying_fitter.py
│   │   ├── aalen_additive_fitter.py
│   │   ├── aalen_johansen_fitter.py
│   │   ├── weibull_fitter.py, weibull_aft_fitter.py
│   │   ├── log_normal_fitter.py, log_normal_aft_fitter.py
│   │   ├── log_logistic_fitter.py, log_logistic_aft_fitter.py
│   │   ├── generalized_gamma_fitter.py
│   │   ├── generalized_gamma_regression_fitter.py
│   │   ├── piecewise_exponential_fitter.py
│   │   ├── piecewise_exponential_regression_fitter.py
│   │   ├── spline_fitter.py
│   │   ├── crc_spline_fitter.py
│   │   ├── mixture_cure_fitter.py
│   │   ├── npmle.py               # Non-parametric MLE
│   │   └── mixins.py              # Shared functionality
│   ├── utils/                     # Utilities (~2K lines)
│   │   ├── __init__.py            # Data manipulation, conversions
│   │   ├── concordance.py         # C-index calculations
│   │   ├── btree.py               # Binary tree structures
│   │   ├── lowess.py              # LOWESS smoothing
│   │   └── printer.py             # Formatted output
│   ├── datasets/                  # Sample datasets (28+ CSV/DAT files)
│   │   └── __init__.py            # Dataset loader functions
│   ├── cmprsk/                    # R package for competing risks (dev branch only)
│   │   ├── R/                     # R source code
│   │   ├── src/                   # Fortran source code
│   │   ├── man/                   # R documentation files
│   │   ├── tests/                 # R tests
│   │   ├── DESCRIPTION            # R package metadata
│   │   └── NAMESPACE              # R package exports
│   ├── tests/                     # Test suite (9 modules)
│   │   ├── test_estimation.py    # Main fitter tests
│   │   ├── test_statistics.py
│   │   ├── test_plotting.py
│   │   ├── test_npmle.py
│   │   ├── test_generate_datasets.py
│   │   └── utils/                # Utility tests
│   ├── statistics.py              # Statistical tests (~975 lines)
│   ├── plotting.py                # Visualization (~1K lines)
│   ├── calibration.py             # Model calibration
│   ├── generate_datasets.py       # Synthetic data generation
│   ├── exceptions.py              # Custom exceptions
│   └── version.py                 # Version string
├── docs/                          # Sphinx documentation
│   ├── conf.py                    # Sphinx configuration
│   ├── jupyter_notebooks/         # Tutorial notebooks
│   └── *.rst                      # Documentation pages
├── examples/                      # Example scripts
├── experiments/                   # Research experiments
├── perf_tests/                    # Performance benchmarks
├── reqs/                          # Requirements files
│   ├── base-requirements.txt      # Core dependencies
│   ├── dev-requirements.txt       # Development tools
│   └── doc-requirements.txt       # Documentation build
├── .github/                       # GitHub configuration
│   ├── workflows/                 # CI/CD pipelines
│   │   ├── ci.yaml                # Main test pipeline
│   │   └── pythonpublish.yml      # PyPI publishing
│   └── CONTRIBUTING.md            # Contribution guidelines
├── setup.py                       # Package configuration
├── conftest.py                    # Pytest configuration
├── Makefile                       # Development shortcuts
├── .pre-commit-config.yaml        # Pre-commit hooks
├── mypy.ini                       # Type checking config
├── .prospector.yaml               # Code quality config
└── README.md, CHANGELOG.md, LICENSE

```

## Key Modules and Components

### 1. Fitters Module (`lifelines/fitters/`)

The heart of the library - contains all survival analysis models organized by type.

#### Class Hierarchy

```
BaseFitter (root class)
├── UnivariateFitter (single variable models)
│   ├── NonParametricUnivariateFitter
│   │   ├── KaplanMeierFitter
│   │   ├── NelsonAalenFitter
│   │   ├── BreslowFlemingHarringtonFitter
│   │   └── AalenJohansenFitter (competing risks)
│   └── ParametricUnivariateFitter
│       ├── WeibullFitter
│       ├── ExponentialFitter
│       ├── LogNormalFitter
│       ├── LogLogisticFitter
│       └── GeneralizedGammaFitter
└── RegressionFitter (regression models)
    ├── SemiParametricRegressionFitter
    │   ├── CoxPHFitter (3.2K lines - proportional hazards)
    │   ├── CoxTimeVaryingFitter (time-varying covariates)
    │   └── AalenAdditiveFitter (additive hazards)
    └── ParametricRegressionFitter
        ├── ParametricAFTRegressionFitter (AFT models)
        │   ├── WeibullAFTFitter
        │   ├── LogNormalAFTFitter
        │   └── LogLogisticAFTFitter
        ├── GeneralizedGammaRegressionFitter
        ├── PiecewiseExponentialRegressionFitter
        ├── SplineFitter (spline-based hazards)
        ├── CRCSplineFitter (Crowther-Royston-Clements)
        └── MixtureCureFitter (cure models)
```

#### Mixins

- **`SplineFitterMixin`**: Provides spline basis function generation (used by SplineFitter, CRCSplineFitter)
- **`ProportionalHazardMixin`**: Adds PH assumption testing (used in CoxPHFitter)

#### Key Files

- **`fitters/__init__.py`**: Base classes (`BaseFitter`, `UnivariateFitter`, `RegressionFitter`, etc.)
- **`fitters/coxph_fitter.py`**: Cox proportional hazards model (most complex, 3.2K lines)
- **`fitters/npmle.py`**: Non-parametric maximum likelihood estimation for interval-censored data

### 2. Utils Module (`lifelines/utils/`)

Rich set of data manipulation and statistical utilities:

**Core Functions:**
- `concordance_index()` - Model performance metric (C-index)
- `qth_survival_times()` - Calculate quantile survival times
- `median_survival_times()` - Extract median survival
- `survival_table_from_events()` - Build life tables from event data
- `to_long_format()`, `to_episodic_format()` - Data format conversions
- `datetimes_to_durations()` - Convert datetime ranges to durations
- `add_covariate_to_timeline()` - Merge time-varying covariates
- `k_fold_cross_validation()` - Model validation
- `find_best_parametric_model()` - Automated model selection

**Submodules:**
- `utils/concordance.py` - C-index computation (303 lines)
- `utils/btree.py` - Binary tree structures for efficient lookups
- `utils/lowess.py` - LOWESS smoothing implementation
- `utils/printer.py` - Formatted output utilities (191 lines)

### 3. Statistics Module (`lifelines/statistics.py`)

Hypothesis testing and power analysis:

- `logrank_test()` - Compare survival curves between groups
- `multivariate_logrank_test()` - Multiple group comparisons
- `pairwise_logrank_test()` - All pairwise comparisons
- `survival_difference_at_fixed_point_in_time_test()` - Point-in-time comparisons
- `proportional_hazard_test()` - Test proportional hazards assumption
- `power_under_cph()` - Power calculations for Cox models
- `sample_size_necessary_under_cph()` - Sample size determination

**Returns:** `StatisticalResult` objects with test statistics, p-values, and metadata

### 4. Plotting Module (`lifelines/plotting.py`)

Visualization utilities with matplotlib integration:

- `add_at_risk_counts()` - Add risk tables below survival plots
- `plot_lifetimes()` - Individual lifetime trajectory plots
- `plot_interval_censored_lifetimes()` - For interval-censored data
- `qq_plot()` - Quantile-quantile plots for model assessment
- `cdf_plot()` - Empirical vs model CDF comparison
- `rmst_plot()` - Restricted mean survival time plots
- `loglogs_plot()` - Log-log transformation plots (test PH assumption)

### 5. Datasets Module (`lifelines/datasets/`)

Real-world datasets for examples and testing:

**Loader Functions:**
- `load_waltons()` - Classic survival dataset
- `load_rossi()` - Recidivism data
- `load_kidney_transplant()` - Medical dataset
- `load_lung()`, `load_larynx()` - Cancer studies
- `load_regression_dataset()` - Simulated regression data
- `load_multicenter_aids_cohort_study()` - AIDS cohort
- `load_diabetes()` - Interval-censored diabetes data

**Files:** 28+ CSV/DAT files in `lifelines/datasets/` directory

## Development Setup

### Initial Setup

```bash
# Clone and navigate to repository
cd /home/user/lifelines

# Create/activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install using Makefile (installs dev requirements + pre-commit)
make init

# OR manually install
pip install -e .
pip install -r reqs/dev-requirements.txt
pre-commit install
```

### Available Make Targets

```bash
make init          # Setup dev environment (install deps + pre-commit hooks)
make test          # Run tests with coverage
make lint          # Run prospector + black formatting
make black         # Format code with black (line length 120)
make check_format  # Check code formatting without changing files
make pre           # Run all pre-commit hooks
```

## Testing Conventions

### Running Tests

```bash
# Run all tests (verbose)
pytest lifelines/tests/ -vv

# Run with coverage
pytest lifelines/ -rfs --cov=lifelines --block=False --cov-report term-missing

# Run specific test file
pytest lifelines/tests/test_estimation.py -vv

# Run specific test
pytest lifelines/tests/test_estimation.py::test_kaplan_meier -vv
```

### Test Structure

- **Main test file:** `lifelines/tests/test_estimation.py` (comprehensive fitter tests)
- **Random seed management:** `conftest.py` sets random seed before each test for reproducibility
- **Fixtures:** Common fixtures for sample data (lifetimes, waltons dataset)
- **Custom pytest options:** `--block` flag controls plot blocking

### Testing Patterns

```python
# Example test structure
def test_some_functionality():
    # Use dataset loaders
    from lifelines.datasets import load_waltons
    df = load_waltons()

    # Create fitter
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()

    # Fit model
    kmf.fit(df['T'], df['E'])

    # Assertions using pandas/numpy testing
    from pandas.testing import assert_frame_equal
    from numpy.testing import assert_allclose

    assert_allclose(kmf.median_survival_time_, expected_value, rtol=1e-3)
```

### Test Configuration

- **`conftest.py`**: Random seed management, custom pytest options
- **Flaky tests:** Use `@flaky` decorator for non-deterministic tests
- **Coverage config:** `.coveragerc` defines coverage settings

## Code Style and Formatting

### Formatting Rules

**Tool:** Black (line length varies by context)
- **Pre-commit hook:** Line length 130
- **Manual formatting:** Line length 120
- **Command:** `black lifelines/ -l 120` or `make black`

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- Trailing whitespace removal
- AST syntax checking
- YAML validation
- End-of-file fixes
- Encoding pragma fixes
- Mixed line ending fixes
- Black formatting (line length 130)

**Install hooks:** `pre-commit install` (or `make init`)
**Run manually:** `pre-commit run --all-files` (or `make pre`)

### Type Checking

- **Tool:** mypy (configuration in `mypy.ini`)
- Not enforced in CI but recommended for new code

### Code Quality

- **Tool:** Prospector (configuration in `.prospector.yaml`)
- **Run:** `make lint` or `prospector --output-format grouped`

## Coding Conventions and Patterns

### Naming Conventions

1. **Fitted attributes end with underscore:**
   ```python
   self.survival_function_     # Fitted survival function
   self.cumulative_hazard_     # Fitted cumulative hazard
   self.confidence_interval_   # Confidence intervals
   self.median_survival_time_  # Median survival time
   ```

2. **Private methods start with underscore:**
   ```python
   def _preprocess_inputs(self, ...)
   def _additive_estimate(self, ...)
   def _create_initial_point(self, ...)
   ```

3. **Class names are descriptive:**
   ```python
   KaplanMeierFitter
   CoxPHFitter
   WeibullAFTFitter
   ```

### Design Patterns

#### 1. Decorator Pattern for Censoring Types

```python
from lifelines.fitters import CensoringType

@CensoringType.right_censoring
def fit(self, durations, event_observed=None, ...):
    # Method automatically marked as supporting right censoring
    pass
```

Types: `@CensoringType.right_censoring`, `@CensoringType.left_censoring`, `@CensoringType.interval_censoring`

#### 2. Mixin Pattern

```python
class SplineFitter(SplineFitterMixin, ParametricRegressionFitter):
    # Inherits spline basis functions from SplineFitterMixin
    # Inherits regression framework from ParametricRegressionFitter
    pass
```

#### 3. Template Method Pattern

Base classes define workflow, subclasses implement specifics:

```python
class ParametricRegressionFitter(RegressionFitter):
    def fit(self, df, duration_col, event_col=None, ...):
        # Common preprocessing
        self._preprocess_inputs(...)

        # Subclass-specific logic
        self._fit_model(...)

        # Common post-processing
        self._compute_confidence_intervals(...)
        return self
```

### Common Code Patterns

#### Input Validation

```python
def fit(self, durations, event_observed=None, ...):
    # Check for NaNs/Infs
    check_nans_or_infs(durations)

    # Preprocess and convert to proper types
    self.durations = np.asarray(durations)
    self.event_observed = np.asarray(event_observed) if event_observed is not None else np.ones_like(durations)
```

#### Numerical Safety

```python
from lifelines.utils import safe_exp

# Instead of np.exp() which can overflow
hazard = safe_exp(log_hazard)
```

#### DataFrame Operations

```python
# Lifelines extensively uses pandas DataFrames
import pandas as pd

# Create survival function DataFrame
survival_function = pd.DataFrame({
    'survival': survival_values,
}, index=timeline)
```

### Documentation Patterns

#### Docstring Style

```python
def logrank_test(durations_A, durations_B, event_observed_A, event_observed_B, **kwargs):
    """
    Measures and reports on whether two intensity processes are different.

    Parameters
    ----------
    durations_A: iterable
        a (n,) array of event durations (birth to death,...) for the first population.
    durations_B: iterable
        a (n,) array of event durations (birth to death,...) for the second population.
    event_observed_A: iterable
        a (n,) array of censorship flags, 1 if observed, 0 if not, for the first population.
    event_observed_B: iterable
        a (n,) array of censorship flags, 1 if observed, 0 if not, for the second population.

    Returns
    -------
    StatisticalResult
        a StatisticalResult object with properties 'p_value', 'test_statistic', 'summary', etc.

    See Also
    --------
    multivariate_logrank_test
    pairwise_logrank_test

    Examples
    --------
    >>> from lifelines.statistics import logrank_test
    >>> from lifelines.datasets import load_waltons
    >>> df = load_waltons()
    >>> result = logrank_test(df.loc[df['group']=='control', 'T'],
    ...                       df.loc[df['group']=='treatment', 'T'],
    ...                       df.loc[df['group']=='control', 'E'],
    ...                       df.loc[df['group']=='treatment', 'E'])
    >>> result.p_value
    0.0034...
    """
```

## Dependencies

### Core Requirements (`reqs/base-requirements.txt`)

```
numpy >= 1.14.0          # Numerical computations
scipy >= 1.7.0           # Scientific computing, optimization
pandas >= 2.1            # Data manipulation
matplotlib >= 3.0        # Plotting
autograd >= 1.5          # Automatic differentiation
autograd-gamma >= 0.3    # Gamma function for autograd
formulaic >= 0.2.2       # R-style formulas
```

### Development Requirements

```
pytest >= 4.6            # Testing framework
pytest-cov               # Coverage reporting
black                    # Code formatting
pre-commit               # Git hooks
prospector               # Code quality
mypy                     # Type checking
statsmodels              # Additional statistical tests
flaky                    # Flaky test handling
```

### Documentation Requirements

```
sphinx == 7.2.6          # Documentation generation
sphinx_rtd_theme == 2.0.0  # ReadTheDocs theme
nbsphinx == 0.9.3        # Jupyter notebook integration
```

## CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/ci.yaml`)

**Triggers:** Pull requests, pushes, manual workflow dispatch

**Matrix Testing:**
- **OS:** ubuntu-latest
- **Python versions:** 3.9, 3.10, 3.11, 3.12
- **Strategy:** fail-fast enabled

**Steps:**
1. Checkout source
2. Setup Python version
3. Install package in editable mode: `pip install -e .`
4. Install dev requirements: `pip install -r reqs/dev-requirements.txt`
5. Run tests: `pytest lifelines/tests/ -vv`

### Publishing Workflow (`.github/workflows/pythonpublish.yml`)

Publishes to PyPI on new releases

## Important File Locations

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `setup.py` | Package configuration | `/home/user/lifelines/setup.py` |
| `conftest.py` | Pytest configuration | `/home/user/lifelines/conftest.py` |
| `.pre-commit-config.yaml` | Pre-commit hooks | `/home/user/lifelines/.pre-commit-config.yaml` |
| `mypy.ini` | Type checking config | `/home/user/lifelines/mypy.ini` |
| `.prospector.yaml` | Code quality config | `/home/user/lifelines/.prospector.yaml` |
| `.readthedocs.yaml` | ReadTheDocs config | `/home/user/lifelines/.readthedocs.yaml` |
| `Makefile` | Development shortcuts | `/home/user/lifelines/Makefile` |

### Documentation Files

| File | Purpose | Location |
|------|---------|----------|
| `README.md` | Project overview | `/home/user/lifelines/README.md` |
| `CHANGELOG.md` | Version history (80KB+) | `/home/user/lifelines/CHANGELOG.md` |
| `CONTRIBUTING.md` | Contribution guide | `/home/user/lifelines/.github/CONTRIBUTING.md` |
| `LICENSE` | MIT License | `/home/user/lifelines/LICENSE` |

### Version Information

- **Version file:** `lifelines/version.py`
- **Format:** `__version__ = "X.Y.Z"`
- **Current:** 0.30.0

## Tips for AI Assistants

### When Working on This Codebase

1. **Always run tests after changes:**
   ```bash
   pytest lifelines/tests/ -vv
   ```

2. **Format code before committing:**
   ```bash
   make black
   # or
   black lifelines/ -l 120
   ```

3. **Understand the class hierarchy:**
   - Check `lifelines/fitters/__init__.py` for base classes
   - Most fitters inherit from `BaseFitter` and add specific functionality
   - Fitted attributes always end with `_`

4. **Use existing patterns:**
   - Look at similar fitters for patterns (e.g., other AFT models)
   - Follow the template method pattern in base classes
   - Use decorators for censoring type markers

5. **Data handling:**
   - Validate inputs with `check_nans_or_infs()`
   - Convert to numpy arrays for computation
   - Return pandas DataFrames for user-facing results
   - Use `safe_exp()` for numerical stability

6. **Testing:**
   - Add tests to appropriate test file (usually `test_estimation.py`)
   - Use dataset loaders from `lifelines.datasets`
   - Check both statistical correctness and edge cases
   - Test with different censoring types if applicable

7. **Documentation:**
   - Add comprehensive docstrings with examples
   - Include parameter types and descriptions
   - Add "See Also" section for related functions
   - Use proper reStructuredText formatting for Sphinx

8. **Mathematical operations:**
   - Use `autograd` for automatic differentiation
   - Use `scipy.optimize` for optimization problems
   - Leverage `scipy.stats` for distributions
   - Be mindful of numerical stability (overflow/underflow)

### Common Gotchas

1. **Line length:** Pre-commit uses 130, manual black uses 120 - be consistent
2. **Random seeds:** Tests use random seeds from `conftest.py` for reproducibility
3. **Fitted attributes:** Always append `_` to fitted attributes (scikit-learn convention)
4. **DataFrame indexing:** Survival functions and hazards use time as index
5. **Censoring indicators:** 1 = event observed, 0 = censored (standard convention)
6. **Import paths:** Main classes exported from `lifelines/__init__.py`, use those for examples

### File Size Reference

When editing large files, be aware of their size:
- `fitters/coxph_fitter.py` - 3,282 lines (largest single fitter)
- `fitters/__init__.py` - 3,638 lines (base classes)
- `statistics.py` - 975 lines
- `plotting.py` - 1,070 lines
- `utils/__init__.py` - 1,957 lines

### Supported Python Versions

- **Minimum:** Python 3.9
- **Tested:** 3.9, 3.10, 3.11, 3.12
- **CI:** All versions tested on ubuntu-latest

## Dev Branch: cmprsk R Package for Competing Risks

**Note:** This section applies only to the `dev` branch.

The `dev` branch contains an R package `cmprsk` in the `lifelines/cmprsk/` directory. This is a reference implementation of competing risks analysis methods in R.

### Package Overview

**cmprsk** (Competing Risks) is an R package version 2.2-12 by Bob Gray that provides:
- Estimation, testing, and regression modeling of subdistribution functions in competing risks
- Implementation of methods described in two seminal papers:
  - **Gray (1988)**: "A class of K-sample tests for comparing the cumulative incidence of a competing risk" (Annals of Statistics)
  - **Fine & Gray (1999)**: "A proportional hazards model for the subdistribution of a competing risk" (JASA)

### Directory Structure

```
lifelines/cmprsk/
├── R/
│   └── cmprsk.R              # Main R source code (534 lines)
├── src/                      # Fortran source code for performance
│   ├── crr.f                 # CRR model implementation (577 lines)
│   ├── cincsub.f             # Cumulative incidence subroutines (105 lines)
│   ├── crstm.f               # Competing risks test statistics (256 lines)
│   └── tpoi.f                # Time point interpolation (34 lines)
├── man/                      # R documentation (.Rd files)
│   ├── crr.Rd                # Competing risks regression docs
│   ├── cuminc.Rd             # Cumulative incidence docs
│   ├── predict.crr.Rd        # Prediction method docs
│   ├── summary.crr.Rd        # Summary method docs
│   ├── plot.cuminc.Rd        # Plotting method docs
│   └── ...                   # Other documentation files
├── tests/
│   ├── test.R                # R test script
│   ├── test.Rout.save        # Expected test output
│   ├── Rplots.pdf            # Test plots (PDF)
│   └── Rplots.ps             # Test plots (PostScript)
├── DESCRIPTION               # Package metadata
├── NAMESPACE                 # Package exports and imports
├── COPYING                   # GPL license (≥2)
└── MD5                       # MD5 checksums
```

### Main Functions

#### 1. **crr()** - Competing Risks Regression

Fits the Fine-Gray proportional subdistribution hazards model.

**Key Parameters:**
- `ftime`: Failure/censoring times
- `fstatus`: Status indicator (unique codes for each failure type + censoring)
- `cov1`: Fixed covariates matrix
- `cov2`: Time-varying covariates (multiplied by time functions)
- `tf`: Time function for cov2
- `cengroup`: Censoring group indicator
- `failcode`: Code for failure type of interest (default: 1)
- `cencode`: Code for censored observations (default: 0)

**Implementation Details:**
- Uses Newton-Raphson optimization
- Calls Fortran subroutines for numerical computation:
  - `crrfsv`: Function, score, and variance computation
  - `crrf`: Function evaluation only
  - `crrvv`: Variance-covariance matrix
  - `crrsr`: Score residuals
  - `crrfit`: Baseline subdistribution hazard jumps
- Supports time-varying effects via `cov2` and `tf` parameters
- Estimates censoring distribution separately within cengroups using Kaplan-Meier

**Returns:** `crr` object with:
- `coef`: Regression coefficients
- `loglik`: Log pseudo-likelihood
- `var`: Variance-covariance matrix
- `res`: Score residuals
- `uftime`: Unique failure times
- `bfitj`: Baseline hazard jumps (for predictions)
- `converged`: Convergence indicator

#### 2. **cuminc()** - Cumulative Incidence Analysis

Estimates cumulative incidence functions and performs tests across groups.

**Key Parameters:**
- `ftime`: Failure times
- `fstatus`: Failure type codes
- `group`: Group variable for comparisons
- `strata`: Stratification variable for tests
- `rho`: Power of weight function (default: 0)
- `cencode`: Censoring code (default: 0)

**Implementation Details:**
- Calls Fortran subroutine `cinc` for cumulative incidence estimation
- Uses `crstm` for test statistics (Gray's K-sample test)
- Estimates computed separately for each group × cause combination
- Tests compare subdistributions across groups using weighted statistics

**Returns:** List with:
- Components for each group-cause combination containing:
  - `time`: Time points
  - `est`: Cumulative incidence estimates
  - `var`: Variance estimates
- `Tests`: Chi-square test statistics and p-values (if multiple groups)

#### 3. Supporting Functions

- **`predict.crr()`**: Estimate subdistributions for new covariate values
- **`summary.crr()`**: Detailed regression results with confidence intervals
- **`plot.cuminc()`**: Visualize cumulative incidence curves
- **`plot.predict.crr()`**: Plot predicted subdistributions
- **`timepoints()`**: Extract estimates at specific time points
- **`print.crr()`**, **`print.cuminc()`**: Print methods

### Fortran Source Code

The package uses Fortran for computational efficiency:

1. **crr.f** (577 lines): Core CRR model computations
   - Function evaluation
   - Score vector and information matrix
   - Variance-covariance estimation
   - Score residuals
   - Baseline hazard jumps

2. **cincsub.f** (105 lines): Cumulative incidence subroutine
   - Non-parametric CIF estimation
   - Variance calculations using Aalen's method

3. **crstm.f** (256 lines): Competing risks test statistics
   - Implements Gray's K-sample test
   - Stratified tests
   - Weighted statistics

4. **tpoi.f** (34 lines): Time point interpolation utility

### Dependencies

- **R** (≥ 3.0.0)
- **survival** package (for Kaplan-Meier censoring distribution estimation)

### Purpose in lifelines Repository

This R package serves as a **reference implementation** for competing risks methods. It can be used to:

1. **Validate** Python implementations of competing risks models
2. **Compare** results between R and Python
3. **Benchmark** performance and numerical accuracy
4. **Reference** algorithmic details from the original implementation
5. **Test** edge cases and verify behavior

### Integration Notes

**Important:** The cmprsk package is written in R and Fortran, not Python. It is **not directly integrated** into the lifelines Python package. Instead, it exists as:

- A reference for implementing similar methods in Python
- A validation tool for comparing results
- Documentation of the canonical implementation

If you need to run or test this R package:

```bash
# Install R (if not already installed)
# Build the package
R CMD build lifelines/cmprsk/

# Install the package
R CMD INSTALL cmprsk_2.2-12.tar.gz

# Or install in R
install.packages("cmprsk")  # from CRAN

# Load and use
library(cmprsk)
?crr
?cuminc
```

### Key References

1. **Gray RJ (1988)** "A class of K-sample tests for comparing the cumulative incidence of a competing risk." *Annals of Statistics* 16:1141-1154. DOI: 10.1214/aos/1176350951

2. **Fine JP and Gray RJ (1999)** "A proportional hazards model for the subdistribution of a competing risk." *Journal of the American Statistical Association* 94:496-509. DOI: 10.1080/01621459.1999.10474144

3. **Aalen O (1978)** "Nonparametric estimation of partial transition probabilities in multiple decrement models." *Annals of Statistics* 6:534-545.

### For AI Assistants

When working with competing risks in the lifelines Python codebase:

1. **Reference the R implementation** to understand algorithm details
2. **Do not modify** the cmprsk R package - it's a reference implementation
3. **Compare outputs** when implementing Python equivalents
4. **Check numerical accuracy** against the Fortran/R results
5. **Study the Fortran code** for optimization techniques and numerical methods

The R package is licensed under **GPL (≥ 2)**, while lifelines itself is MIT licensed. Be mindful of licensing implications if adapting code.

## Contributing Workflow

1. **Open an issue** to discuss changes before implementing
2. **Fork and branch** from the appropriate base branch
3. **Make changes** following code style guidelines
4. **Add tests** for new functionality
5. **Format code** with black (line length 120-130)
6. **Run tests** to ensure everything passes
7. **Submit PR** with clear description of changes

## Additional Resources

- **Documentation:** http://lifelines.readthedocs.org/
- **Discussions:** https://github.com/CamDavidsonPilon/lifelines/discussions
- **Stack Exchange:** https://stats.stackexchange.com/ (search "lifelines")
- **Issues:** https://github.com/camdavidsonpilon/lifelines/issues

---

**Note for AI Assistants:** This document is maintained to help AI systems understand the lifelines codebase structure and conventions. When in doubt, examine existing code for patterns and consult the documentation at the URLs above.
