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
