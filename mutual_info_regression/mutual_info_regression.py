import numpy as np # type: ignore
import numpy.typing as npt # type: ignore
from sklearn.feature_selection import mutual_info_regression as sklearn_mi # type: ignore

def _subsample(
        X, 
        y, 
        k:int,
        splits:np.ndarray,
        **kwargs
        ) -> np.ndarray:
    
    n_samples, _ = X.shape
    MIs = []

    for i in range(len(splits)):
        a = np.random.permutation(n_samples)
        split = splits[i]
        boundaries = np.linspace(0, n_samples, split + 1, dtype=int)

        MI_T = np.zeros(split)

        for j in range(split):
            _X = X[a[boundaries[j]:boundaries[j + 1]], :]
            _y = y[a[boundaries[j]:boundaries[j + 1]]]
            if _X.shape[0] <= k:
                continue

            MI_vals = sklearn_mi(
                _X,
                _y,
                discrete_features=False,
                n_neighbors=k,
                **kwargs
            )
            MI_T[j] = np.mean(MI_vals)

        MIs.append((split, MI_T / np.log(2))) 
    return MIs

def _std(MIs, ddof:int=1) -> float:

    splits = np.asarray([row[0] for row in MIs], dtype=int)
    MI_vectors = [np.asarray(row[1], dtype=float) for row in MIs]

    vars_ = np.array([
        np.var(vec, ddof=ddof if vec.size > ddof else 0)
        for vec in MI_vectors[1:]
    ])
    k = splits[1:]
    var_hat = np.sum(((k - 1) / k) * vars_) / np.sum(k - 1)

    return float(np.sqrt(var_hat))

def mutual_info_regression(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    k: list[int],
    splits: list[int],
    **kwargs
) -> dict:
    """
    Compute the mutual information between each feature and the target variable using the Holmes and Nemenman (2019) adaptation to the Kraskov estimator

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The target variable.
    k : list of int
        The number of neighbors to use for estimating the mutual information.
    splits : list of int
        The number of splits to use for subsampling.
    **kwargs : keyword arguments
        Additional arguments to pass to the sklearn mutual_info_regression function.
    Returns
    -------
    means : dict
        A dictionary containing the mean mutual information for each feature.
    error : dict
        A dictionary containing the standard deviation of the mutual information for each feature.
    """
    
    means = {}
    error = {}

    for ks in k:
        mi = _subsample(X, y, ks, splits, **kwargs)
        mi_all = np.concatenate([row[1] for row in mi])

        means[ks] = float(np.mean(mi_all))
        error[ks] = float(_std(mi))

    return {
        'means': means, 
        'error': error,
    }