import itertools

import torch


def generate_permutations(idx_equiv_atoms: list[list[int]]) -> torch.Tensor:
    """

    Parameters
    ----------
    idx_equiv_atoms : list[list[int]]
        List containing lists of indices for equivalent atoms

    Returns
    -------
    torch.Tensor
        Tensor of all possible permutations

    Example
    -------
    For the reaction between N2 and H3+, the nitrogen atoms have indices 0 and 1,
    while the hydrogen atoms have indices 2, 3, and 4.

    The `idx_equiv_atoms` list should look like [[0,1],[2,3,4]].
    """
    all_perms = []
    for group in idx_equiv_atoms:
        all_perms.append([list(p) for p in itertools.permutations(group)])

    perms = [list(tup) for tup in itertools.product(*all_perms)]

    perms = [sum((sublist for sublist in item), []) for item in perms]

    perms = torch.tensor(perms)

    return perms


def generate_unique_distances(num_atoms: int, idx_equiv_atoms: list[list[int]]) -> int:
    """

    Parameters
    ----------
    num_atoms: int
        The total number of atoms in the system
    idx_equiv_atoms: list[list[int]]
        List of lists representing the groups of permutationally invariant atoms

    Returns
    -------
    num_unique_dist: int
        The number of unique distances in the system taking into account permutational invariance

    Examples
    --------
    The H2O system contains two permutationally invariant hydrogen atoms H1 and H2. The energy is invariant to
    the permutation of the distances O-H1 and O-H2.
    Therefore there are 2 unique distances in the system: O-H and H-H.

    The general formula is,
                                unique distances = n(n-1)/2 + k,
    where n is the number of atom groups and k is the number of groups containing more than a single atom.
    """

    num_perminv_groups = len(idx_equiv_atoms)
    num_reg_groups = num_atoms - sum([len(i) for i in idx_equiv_atoms])
    num_groups = num_reg_groups + num_perminv_groups

    num_unique_dist = (num_groups * (num_groups - 1)) // 2 + num_perminv_groups

    return num_unique_dist
