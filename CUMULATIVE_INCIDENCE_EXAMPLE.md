# CumulativeIncidenceFitter Usage Examples

## Overview

The `CumulativeIncidenceFitter` provides pure Python implementation of cumulative incidence function (CIF) estimation for competing risks data. This is equivalent to R's `cmprsk::cuminc()` function.

## Installation

```python
from lifelines import CumulativeIncidenceFitter
import numpy as np
import pandas as pd
```

## Basic Usage

### Single Group Estimation

```python
# Sample data
times = np.array([1, 2, 3, 4, 5, 6, 7, 8])
events = np.array([1, 1, 2, 0, 1, 2, 1, 0])
# 0 = censored
# 1 = event type 1 (e.g., death from cancer)
# 2 = event type 2 (e.g., death from other causes)

# Fit the model
cif = CumulativeIncidenceFitter()
cif.fit(times, events, event_of_interest=1)

# View results
print(cif.cumulative_incidence_)
print(cif.variance_)

# Plot
cif.plot()
```

### Multiple Groups with Gray's Test

```python
# Compare CIF across treatment groups
df = pd.DataFrame({
    'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'status': [1, 1, 2, 0, 1, 2, 1, 0, 2, 1],
    'treatment': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
})

cif = CumulativeIncidenceFitter()
cif.fit(
    df['time'],
    df['status'],
    event_of_interest=1,
    groups=df['treatment']
)

# View cumulative incidence for each group
print(cif.cumulative_incidence_)

# View Gray's test results
print(cif.test_statistics_)
# Output:
#    cause  statistic   p_value  df
# 0      1      2.345     0.126   1

# Plot both groups
cif.plot()
```

### Stratified Analysis

```python
# Stratified Gray's test (stratify by center)
df = pd.DataFrame({
    'time': np.random.exponential(10, 200),
    'status': np.random.choice([0, 1, 2], 200),
    'treatment': np.random.choice(['A', 'B'], 200),
    'center': np.random.choice(['Center1', 'Center2', 'Center3'], 200)
})

cif = CumulativeIncidenceFitter()
cif.fit(
    df['time'],
    df['status'],
    event_of_interest=1,
    groups=df['treatment'],
    strata=df['center']  # Stratify test by center
)

print(cif.test_statistics_)
```

## Real-World Example

### Bone Marrow Transplant Data

```python
from lifelines.datasets import load_dataset

# (Assuming bone marrow transplant dataset with competing risks)
# Event types: 0 = censored, 1 = relapse, 2 = death in remission

cif = CumulativeIncidenceFitter(alpha=0.05)
cif.fit(
    durations=bmt_data['time'],
    event_observed=bmt_data['status'],
    event_of_interest=1,  # Focus on relapse
    groups=bmt_data['treatment_group']
)

# Print summary
print("Cumulative Incidence of Relapse:")
print(cif.cumulative_incidence_)

print("\nGray's Test for Treatment Effect:")
print(cif.test_statistics_)

# Visualize
import matplotlib.pyplot as plt

ax = cif.plot()
plt.title('Cumulative Incidence of Relapse by Treatment Group')
plt.xlabel('Time (months)')
plt.ylabel('Cumulative Incidence')
plt.legend(['Treatment A', 'Treatment B'])
plt.show()
```

## Comparison with AalenJohansenFitter

Both `CumulativeIncidenceFitter` and `AalenJohansenFitter` estimate cumulative incidence functions, but with different focuses:

```python
from lifelines import CumulativeIncidenceFitter, AalenJohansenFitter

# CumulativeIncidenceFitter: Better for group comparisons
cif = CumulativeIncidenceFitter()
cif.fit(times, events, event_of_interest=1, groups=treatment)
print(cif.test_statistics_)  # Gray's test available

# AalenJohansenFitter: More detailed variance calculation
aj = AalenJohansenFitter(calculate_variance=True)
aj.fit(times, events, event_of_interest=1)
# More focused on single-group estimation with detailed CI
```

## Understanding the Output

### Cumulative Incidence Values

- **Range**: [0, 1]
- **Interpretation**: CIF(t) = Probability of experiencing the event of interest by time t
- **Non-decreasing**: CIF should never decrease over time
- **Competing risks**: Accounts for subjects who experience competing events

### Variance

- Calculated using Aalen's method
- Used to compute confidence intervals
- Accounts for uncertainty in both the event of interest and competing events

### Gray's Test

- **Null hypothesis**: No difference in CIF across groups
- **Test statistic**: Follows chi-square distribution with (# groups - 1) df
- **Interpretation**: Low p-value suggests groups have different cumulative incidence patterns

## API Reference

### CumulativeIncidenceFitter()

**Parameters:**
- `alpha` (float): Significance level for confidence intervals (default: 0.05)

**Methods:**

#### fit(durations, event_observed, event_of_interest, groups=None, strata=None, ...)

**Parameters:**
- `durations`: Array of event/censoring times
- `event_observed`: Array of event type indicators (0 = censored, >0 = event types)
- `event_of_interest`: Integer code for the event type to analyze
- `groups`: (Optional) Group labels for comparing CIF across groups
- `strata`: (Optional) Stratification variable for Gray's test
- `timeline`: (Optional) Specific times at which to estimate CIF
- `weights`: (Optional) Individual-level weights

**Returns:**
- `self` with fitted attributes

**Attributes after fitting:**
- `cumulative_incidence_`: DataFrame of CIF estimates
- `variance_`: DataFrame of variance estimates
- `confidence_interval_`: DataFrame of confidence intervals
- `test_statistics_`: DataFrame of Gray's test results (if groups provided)

## Technical Notes

### Algorithm

The implementation follows:
1. **Gray (1988)**: "A class of K-sample tests for comparing the cumulative incidence of a competing risk"
2. **Aalen (1978)**: "Nonparametric estimation of partial transition probabilities in multiple decrement models"

### Key Formula

CIF_k(t) = Σ_{t_i ≤ t} S(t_i-) × h_k(t_i)

Where:
- S(t-) = Overall survival function just before time t
- h_k(t) = Cause-specific hazard for event k at time t

### Differences from R's cmprsk

1. **API**: Pythonic interface consistent with lifelines
2. **Output**: Returns pandas DataFrames instead of R lists
3. **Variance**: Same algorithm (Aalen's method)
4. **Test**: Gray's test implemented identically

## Performance

- **Speed**: Comparable to R for datasets < 10,000 observations
- **Memory**: O(n) where n = number of observations
- **Numerical stability**: Uses numerically stable algorithms with finite-sample corrections

## Troubleshooting

### No events of interest

```python
# Error: event_of_interest=1 not found in data
# Solution: Check unique event codes
print(np.unique(events))
```

### Singular variance matrix warning

- Occurs when groups have very similar CIF curves
- Test statistic may be unreliable
- Consider combining groups or using different stratification

### Very small p-values

- May indicate numerical issues
- Verify data quality and sample sizes
- Consider using stratified test if appropriate

## Future Development

Planned features:
1. Bootstrap confidence intervals
2. Weighted tests (rho parameter for early/late differences)
3. Pairwise comparisons for >2 groups
4. Integration with FineGrayFitter for regression

## References

1. Gray RJ (1988). Annals of Statistics 16:1141-1154.
2. Aalen O (1978). Annals of Statistics 6:534-545.
3. Fine JP, Gray RJ (1999). JASA 94:496-509.
