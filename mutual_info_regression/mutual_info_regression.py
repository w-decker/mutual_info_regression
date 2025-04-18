import numpy as np
from sklearn.feature_selection import mutual_info_regression


def _subsample(
            X, 
            y, 
            k:int, 
            splits:np.ndarray,
            **kwargs
) -> np.ndarray:
    n_samples, _ = X.shape

    n = len(splits)
    MIs = np.empty(shape=(n,2))
    MIs[:,0] = splits

    for i in np.arange(0, n):

        a = np.random.permutation(n_samples)
        split = splits[i]
        _l = _l = np.linspace(0, n_samples, split + 1, dtype=int)

        MI_T = np.zeros((split, 1))

        for j in np.arange(0, split):
            _X = X[a[_l[j]:_l[j + 1]], :]
            _y = y[a[_l[j]:_l[j + 1]]]
            MI_vals = mutual_info_regression(_X, 
                                             _y, 
                                             discrete_features=False, 
                                             n_neighbors=k, 
                                             **kwargs
                                             )
            MI_T[j] = np.mean(MI_vals)

        MIs[i, 1] = np.divide(MI_T, np.log(2))
        return MIs