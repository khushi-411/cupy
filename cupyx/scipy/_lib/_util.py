import cupy as cp

import numpy as np


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.

    Parameters
    ----------
    a : array-like
        The array-like input
    check_finite : bool, optional
        By default True. To check whether the input matrices contain
        only finite numbers. Disabling may give a performance gain,
        but may result in problems (crashes, non-termination) if the
        inputs do contain infinites or NaNs
    sparse_ok : bool, optional
        By default False. True if cupy sparse matrices are allowed
    objects_ok : bool, optional
        By default False. True if arrays with dype('O') are allowed
    mask_ok : bool, optional
        By default False. True if masked arrays are allowed.
    as_inexact : bool, optional
        By default False. True to convert the input array to a
        cp.inexact dtype

    Returns
    -------
    ret : cupy.ndarray
        The converted validated array

    """
    if not sparse_ok:
        import cupyx.scipy.sparse
        if cupyx.scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the cupyx.scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)

    # TODO: remove these comments when CuPy supports masked arrays
    # Ref Issue: https://github.com/cupy/cupy/issues/2225
    # if not mask_ok:
    #     if cp.ma.isMaskedArray(a):
    #         raise ValueError('masked arrays are not supported')

    toarray = cp.asarray_chkfinite if check_finite else cp.asarray
    a = toarray(a)

    # TODO: remove these comments when CuPy supports 'object' dtype
    # if not objects_ok:
    #    if a.dtype is cp.dtype('O'):
    #        raise ValueError('object arrays are not supported')

    if as_inexact:
        if not np.issubdtype(a.dtype, cp.inexact):
            a = toarray(a, dtype=cp.float_)
    return a
