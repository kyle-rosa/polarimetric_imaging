import math
from pathlib import Path

import torch
from torch import Tensor
import numpy as np


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


def to_numpy_byte_image(
    tensor_image
) -> np.ndarray:
    """
    Converts a torch tensor image to a numpy uint8 array.
    Args:
        tensor_image (torch.Tensor): A torch tensor of a floating point data type.
    Returns:
        np.ndarray: A numpy uint8 array of shape with the same shape as tensor_image.
    """
    return tensor_image.multiply(256).floor().clamp(0, 255).to(torch.uint8).to('cpu').numpy()


def visualise_polarisation(
    angles_dict: dict[str, Tensor],
    stokes: dict[str, Tensor],
    cyclic_colourmap: Tensor,
    gamma: float
):
    # Calculate (real-valued) scalar fields, with ranges in [0, 1].
    scalar_fields = {
        **angles_dict,
        'Stokes I': stokes['I'].div(2),
        'Stokes Q': stokes['Q'].add(1).div(2),
        'Stokes U': stokes['U'].add(1).div(2),
        'Intensity of Polarised Light': stokes['L'].abs(),
        'Intensity of Unpolarised Light': stokes['I'].pow(2).sub(stokes['L'].abs().pow(2)).clamp(0).pow(1 / 2),
        'Degree of Linear Polarisation': stokes['L'].abs().div(stokes['I'].add(1e-8))
    }
    scalar_colourings = {
        k: (
            scalar_field.sub(scalar_field.min())
            .div(scalar_field.max().sub(scalar_field.min()))
            .pow(gamma)[..., None]
            .expand(-1, -1, -1, 3)
        )
        for (k, scalar_field) in scalar_fields.items()
    }
    scalar_colourings = {
        **scalar_colourings,
        **{
            f'Stokes {k}': (
                (normalised_field := stokes[k].div((stokes['Q'].pow(2) + stokes['U'].pow(2)).pow(1/2)))
                .sub(normalised_field.min())
                .div(normalised_field.max().sub(normalised_field.min()))
                .pow(gamma)[..., None]
                .expand(-1, -1, -1, 3)
            )
            for k in ['Q', 'U']
        }
    }

    complex_fields = {
        'L': stokes['L'],
        'ADoLP': stokes['L'].div(stokes['I'].add(1e-8)),
    }
    domain_colourings = {
        k: make_domain_colouring(
            complex_field=v,
            cyclic_colourmap=cyclic_colourmap,
            gamma=gamma,
        ) for (k, v) in complex_fields.items()
    }

    phase = torch.atan2(stokes['U'], stokes['Q']).div(math.tau).add(1 / 2)
    phase_idx = phase.mul(len(cyclic_colourmap)).floor().clamp(0, len(cyclic_colourmap)-1).long()
    hsv_colourings = {
        'HSV': (
            cyclic_colourmap[phase_idx, :]
            .mul(scalar_fields['Degree of Linear Polarisation'][..., None]) 
            .add(1 - scalar_fields['Degree of Linear Polarisation'][..., None])
            .mul(scalar_fields['Stokes I'].pow(gamma)[..., None])
        ),
        'HSV-DoLP': (
            cyclic_colourmap[phase_idx, :]
            .mul(scalar_fields['Degree of Linear Polarisation'][..., None]) 
            .add(
                scalar_fields['Stokes I'].pow(gamma)
                .mul(1 - scalar_fields['Degree of Linear Polarisation'])[..., None]
            )
        ),
    }
    visualisations = {
        **scalar_colourings, 
        **domain_colourings, 
        **hsv_colourings
    }
    return {k: to_numpy_byte_image(v[0].expand(-1, -1, 3)) for (k, v) in visualisations.items()}

