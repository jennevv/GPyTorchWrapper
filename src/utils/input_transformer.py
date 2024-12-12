import torch


def xyz_to_invdist_torch(x: torch.Tensor, index: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    x is a tensor of shape (n, m) where m is the number of individual x, y, z coordinates
    and n is the number of data points.

    The x, y, z coordinates must be ordered as x1, y1, z1, x2, y2, z2, ... , xn, yn, zn

    The final tensor containing the interatomic distances will have the shape (n, m/3) where m/3 is the number of atoms.
    The order of the distances is d01, d02, ..., d12, d13, ..., d(m/3-2)(m/3-1)

    Parameters
    ----------
    x : torch.Tensor
        The input tensor of shape (n, m) where m is the number of individual x, y, z coordinates
    index : bool
        returns unique atom indices per distance

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        The inverse interatomic distances tensor or
        a tuple containing the inverse interatomic distances tensor and the unique atom indices per distance
    """

    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    elif len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)

    n, m = x.shape
    num_atoms = m // 3

    coords = x.reshape(n, num_atoms, 3)

    # Calculate pairwise distances
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)

    # Create a mask to zero out the diagonal (self-distances)
    mask = torch.eye(num_atoms, dtype=torch.bool)
    dist = dist.masked_fill(mask, 0)

    # Upper triangular indices
    triu_indices = torch.triu_indices(num_atoms, num_atoms, offset=1)

    # Get the upper triangular part of the distance matrix
    interdist = dist[:, triu_indices[0], triu_indices[1]]

    if index:
        return torch.pow(interdist, -1), torch.transpose(triu_indices, -1, -2)

    return torch.pow(interdist, -1)
