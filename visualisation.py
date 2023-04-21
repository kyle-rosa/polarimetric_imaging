import math
from pathlib import Path

import torch
from torch import Tensor
import numpy as np


def to_numpy_byte_image(
    tensor_image
) -> np.ndarray:
    """
    Converts a torch tensor image to a numpy uint8 array.

    Args:
        tensor_image (torch.Tensor): A torch tensor of shape (batch_size, num_channels, height, width).

    Returns:
        np.ndarray: A numpy uint8 array of shape (batch_size, height, width, num_channels).
    """
    return tensor_image.multiply(256).floor().clamp(0, 255).to(torch.uint8).to('cpu').numpy()


def load_cyclic_colourmap(compute_device: torch.device):
    return torch.load(Path() / "data/input/cyclic_cmap.pt", map_location=compute_device)


def load_diverging_colourmap(compute_device: torch.device):
    return torch.load(Path() / "data/input/roma.pt", map_location=compute_device)


def load_linear_colourmap(compute_device: torch.device):
    return torch.load(Path() / "data/input/grayC.pt", map_location=compute_device)


def make_domain_colouring(
    complex_field: Tensor,
    cyclic_colourmap: Tensor,
    gamma: float
):
    num_colours = cyclic_colourmap.size(0)
    (radius, angle) = (complex_field.abs(), complex_field.angle())
    turning_number = angle.div(math.tau).add(1 / 2).remainder(1)
    normalised_radius = radius.div(radius.max())
    quantized_angle = turning_number.multiply(num_colours).floor().long()
    angle_colours = cyclic_colourmap[quantized_angle, :]
    image = normalised_radius[..., None].pow(gamma).multiply(angle_colours)
    return image


def visualise_polarisation(
    angles_dict: dict[str, Tensor],
    stokes_dict: dict[str, Tensor],
    cyclic_colourmap: Tensor,
    gamma: float
):
    # scalar_fields = {
    #     **angles_dict,
    #     'I': stokes_dict['I'].div(2),
    #     'Q': stokes_dict['Q'].add(1).div(2),
    #     'U': stokes_dict['U'].add(1).div(2),
    #     'IoPL': stokes_dict['L'].abs(),
    # }
    # scalar_fields = {
    #     **scalar_fields,
    #     'IoUL': stokes_dict['I'].pow(2).sub(scalar_fields['IoPL'].pow(2)).pow(1/2),
    #     'DoLP': scalar_fields['IoPL'].div(stokes_dict['I'].add(1e-8)),
    # }
    # scalar_colourings = {
    #     k: scalar_fields[k].pow(gamma)[..., None].expand(-1, -1, -1, 3)
    #     for k in scalar_fields
    # }
    complex_fields = {
        'L': stokes_dict['L'],
        'ADoLP': stokes_dict['L'].div(stokes_dict['I']),
        # 'AoLP': stokes_dict['L'].div(stokes_dict['L'].abs().add(1e-8)),
    }
    domain_colourings = {
        k: make_domain_colouring(
            complex_field=v,
            cyclic_colourmap=cyclic_colourmap,
            gamma=gamma,
        ) for (k, v) in complex_fields.items()
    }
    visualisations = {
        # **scalar_colourings, 
        **domain_colourings, 
    }
    return visualisations
