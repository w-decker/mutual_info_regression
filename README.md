# mutual_info_regression
Estimate mutual information for a continuous target variable using Holmes and Nemenman (2019) estimator approach. The core function `mutual_info_regression()` is mostly a wrapper around `sklearn.feature_selection.mutual_info_regression()` that has been imbued with additional functionality to reduce bias.

# Installation 

```shell
pip install git+https://github.com/w-decker/mutual_info_regression.git
```

# Usage

```python
import numpy as np

from mutual_info_regression import mutual_info_regression

np.random.seed(2025)
num_points = 1000
correlation = 0.99  # Correlation coefficient

# Generate correlated bivariate Gaussian data
x1 = np.random.normal(0, 1, size=num_points)
x2 = np.random.normal(0, 1, size=num_points)
x3 = correlation * x1 + np.sqrt(1 - correlation**2) * x2

X = x1.reshape(-1, 1)  # Feature matrix
Y = x3                # Target vector

# Analytical mutual information between X and Y (in bits)
mi_analytical = -0.5 * np.log2(1 - correlation**2)
print(f"Correlation: {correlation}")
print(f"Analytical MI: {mi_analytical:.6f} bits")

k_list = [1, 2, 3, 4, 5, 7, 10, 15, 20] # neighbors
split_list = list(range(1, 11)) # splits

# compute MI
mi_result = mutual_info_regression(X, Y, k=k_list, splits=split_list)

mi_estimates = np.array([mi_result['means'][ks] for ks in k_list])
error_bars = np.array([mi_result['error'][ks] for ks in k_list])

# Print results
for ks, mi, err in zip(k_list, mi_estimates, error_bars):
    print(f"k = {ks:2d}:  MI = {mi:.6f} ± {err:.6f} bits")
```
```shell
Correlation: 0.99
Analytical MI: 2.825544 bits
k =  1:  MI = 2.742254 ± 0.061551 bits
k =  2:  MI = 2.716965 ± 0.060462 bits
k =  3:  MI = 2.673892 ± 0.040511 bits
k =  4:  MI = 2.635758 ± 0.044245 bits
k =  5:  MI = 2.591905 ± 0.052512 bits
k =  7:  MI = 2.510555 ± 0.036440 bits
k = 10:  MI = 2.381807 ± 0.044381 bits
k = 15:  MI = 2.179466 ± 0.037548 bits
k = 20:  MI = 1.997547 ± 0.024592 bits
```

# References

This codebase references [Holmes and Nemenman (2019)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.022404) and reproduces their [original MATLAB code](https://github.com/EmoryUniversityTheoreticalBiophysics/ContinuousMIEstimation) into Python. 

```bibtex
@article{PhysRevE.100.022404,
  title = {Estimation of mutual information for real-valued data with error bars and controlled bias},
  author = {Holmes, Caroline M. and Nemenman, Ilya},
  journal = {Phys. Rev. E},
  volume = {100},
  issue = {2},
  pages = {022404},
  numpages = {10},
  year = {2019},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.100.022404},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.100.022404}
}
```