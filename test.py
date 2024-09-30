import numpy as np


def _num_dicke_states(N):
    """
    Calculate the number of Dicke states.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    nds: int
        The number of Dicke states.
    """
    if (not float(N).is_integer()):
        raise ValueError("Number of TLS should be an integer")

    if (N < 1):
        raise ValueError("Number of TLS should be non-negative")

    nds = (N/2 + 1)**2 - (N % 2)/4
    return int(nds)


def _num_dicke_ladders(N):
    """
    Calculate the total number of Dicke ladders in the Dicke space.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    Nj: int
        The number of Dicke ladders.
    """
    Nj = (N+1) * 0.5 + (1-np.mod(N, 2)) * 0.5
    return int(Nj)


def get_blocks(N):
    """
    Calculate the number of cumulative elements at each block boundary.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    blocks: np.ndarray
        An array with the number of cumulative elements at the boundary of
        each block.
    """
    num_blocks = _num_dicke_ladders(N)
    blocks = [i * (N+2-i) for i in range(1, num_blocks+1)]
    return blocks


def j_min(N):
    """
    Calculate the minimum value of j for given N.

    Parameters
    ----------
    N: int
        Number of two-level systems.

    Returns
    -------
    jmin: float
        The minimum value of j for odd or even number of two
        level systems.
    """
    if N % 2 == 0:
        return 0
    else:
        return 0.5


def j_vals(N):
    """
    Get the valid values of j for given N.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    jvals: np.ndarray
        The j values for given N as a 1D array.
    """
    j = np.arange(j_min(N), N/2 + 1, 1)
    return j


def m_vals(j):
    """
    Get all the possible values of m or m1 for given j.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    Returns
    -------
    mvals: np.ndarray
        The m values for given j as a 1D array.
    """
    return np.arange(-j, j+1, 1)


def get_index(N, j, m, m1, blocks):
    """
    Get the index in the density matrix for this j, m, m1 value.

    Parameters
    ----------
    N: int
        The number of two-level systems.

    j, m, m1: float
        The j, m, m1 values.

    blocks: np.ndarray
        An 1D array with the number of cumulative elements at the boundary of
        each block.

    Returns
    -------
    mvals: array
        The m values for given j.
    """
    _k = int(j-m1)
    _k_prime = int(j-m)
    block_number = int(N/2 - j)
    offset = 0
    if block_number > 0:
        offset = blocks[block_number-1]
    i = _k_prime + offset
    k = _k + offset
    return (i, k)


def jmm1_dictionary(N):
    """
    Get the index in the density matrix for this j, m, m1 value.

    The (j, m, m1) values are mapped to the (i, k) index of a block
    diagonal matrix which has the structure to capture the permutationally
    symmetric part of the density matrix. For each (j, m, m1) value, first
    we get the block by using the "j" value and then the addition in the
    row/column due to the m and m1 is determined. Four dictionaries are
    returned giving a map from the (j, m, m1) values to (i, k), the inverse
    map, a flattened map and the inverse of the flattened map.
    """
    jmm1_dict = {}
    jmm1_inv = {}
    jmm1_flat = {}
    jmm1_flat_inv = {}
    nds = _num_dicke_states(N)
    blocks = get_blocks(N)

    jvalues = j_vals(N)
    for j in jvalues:
        mvalues = m_vals(j)
        for m in mvalues:
            for m1 in mvalues:
                i, k = get_index(N, j, m, m1, blocks)
                jmm1_dict[(i, k)] = (j, m, m1)
                jmm1_inv[(j, m, m1)] = (i, k)
                l = nds * i+k
                jmm1_flat[l] = (j, m, m1)
                jmm1_flat_inv[(j, m, m1)] = l
    return [jmm1_dict, jmm1_inv, jmm1_flat, jmm1_flat_inv]


def dicke(N, j, m):
    r"""
    Generate a Dicke state as a pure density matrix in the Dicke basis.

    For instance, the superradiant state given by
    :math:`\lvert  j, m\rangle = \lvert 1, 0\rangle` for N = 2,
    and the state is represented as a density matrix of size (nds, nds) or
    (4, 4), with the (1, 1) element set to 1.


    Parameters
    ----------
    N: int
        The number of two-level systems.

    j: float
        The eigenvalue j of the Dicke state (j, m).

    m: float
        The eigenvalue m of the Dicke state (j, m).

    Returns
    -------
    rho: :class:`.Qobj`
        The density matrix.
    """
    nds = _num_dicke_states(N)
    rho = np.zeros((nds, nds))

    jmm1_dict = jmm1_dictionary(N)[1]

    i, k = jmm1_dict[(j, m, m)]
    rho[i, k] = 1.0
    return rho


if __name__ == "__main__":
    N = 2
    test = jmm1_dictionary(N)
    test2 = dicke(N, 1, 1)
