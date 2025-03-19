from gpytorch.kernels import Kernel
from torch import Tensor

from gpytorchwrapper.src.utils.permutational_invariance import (
    generate_interatomic_distance_indices,
    generate_unique_distances,
    generate_ard_expansion,
    generate_dist_permutations,
)


class PermInvKernel(Kernel):
    """
    This class is a base class for all permutationally invariant kernels.
    It handles the ARD expansion automatically and provides a method to generate the permutations of the distances.

    Parameters
    ----------
    n_atoms: int
        The total number of atoms in the system
    idx_equiv_atoms: list[list[int]]
        List of lists representing the groups of permutationally invariant atoms
    select_dims: Tensor
        The indices of the dimensions to be selected
    ard: bool
        Whether to use ARD or not
    kwargs
        Additional keyword arguments

    Attributes
    ----------
    select_dims: Tensor
        The indices of the dimensions to be selected
    idx_equiv_atoms: list[list[int]]
        List of lists representing the groups of permutationally invariant atoms
    ard: bool
        Whether to use ARD or not
    ard_expansion: list[int]
        The ARD expansion
    permutations: Tensor
        The permutations of the distances
    """

    def __init__(
        self,
        n_atoms: int,
        idx_equiv_atoms: list[list[int]],
        select_dims: list[int] = None,
        ard: bool = False,
        **kwargs,
    ):
        n_dist = n_atoms * (n_atoms - 1) // 2
        distance_idx = generate_interatomic_distance_indices(n_atoms)

        if not ard:
            super().__init__(**kwargs)
            if self.ard_num_dims is not None:
                raise NotImplementedError(
                    "Regular ARD is not supported for LinearxMaternKernelPermInv. Set 'ard=True' instead and specify ard_expansion."
                )
        else:
            ard_num_dims = n_dist if not select_dims else len(select_dims)
            num_unique_distances = generate_unique_distances(
                n_atoms, idx_equiv_atoms
            )  # permutationally unique!

            if select_dims:
                distance_idx = [distance_idx[i] for i in select_dims]
                ard_expansion = generate_ard_expansion(distance_idx, idx_equiv_atoms)
            else:
                ard_expansion = generate_ard_expansion(distance_idx, idx_equiv_atoms)
                if num_unique_distances != len(set(ard_expansion)):
                    raise ValueError(
                        "The permutationally invariant ARD expansion failed."
                        f"Expected number of unique distances {num_unique_distances} != {len(set(ard_expansion))}"
                        f"ARD expansion: {ard_expansion}"
                    )

            super().__init__(ard_num_dims=ard_num_dims, **kwargs)

            permutations = generate_dist_permutations(distance_idx, idx_equiv_atoms)

            self.select_dims = select_dims
            self.idx_equiv_atoms = idx_equiv_atoms
            self.ard = ard
            self.ard_expansion = ard_expansion
            self.permutations = permutations
