import math
import torch
import torch.nn.functional as F
from itertools import product
from pathlib import Path

def make_diffusion_kernel(width, height, time):
    return torch.stack(
        torch.meshgrid(
            [
                torch.fft.fftfreq(width).pow(2),
                torch.fft.fftfreq(height).pow(2)
            ], indexing='xy'
        ), dim=-1
    ).sum(dim=-1).multiply(time).multiply(-4*math.pi**2).add(1).pow(1/2)


def diffuse_stokes_features(stokes, diffusion_kernel):
    diffused_stokes = {
        key: torch.fft.ifftn(
            torch.fft.fftn(value, dim=[-2, -1]).multiply(diffusion_kernel),
            dim=[-2, -1]
        )
        for (key, value) in stokes.items()
    }
    diffused_stokes['I'] = diffused_stokes['I'].real
    return diffused_stokes


def dequantize_frame(byte_frame, bit_depth=12):
    byte_depth_constants = torch.tensor(2).pow(
        torch.arange(byte_frame.shape[-1]).multiply(8)
    ).to(byte_frame.device)
    quantized_frame = byte_frame.long().multiply(byte_depth_constants).sum(dim=-1)#.sub(1).float()
    dequantized_frame = quantized_frame.sub(1).div(2**bit_depth - 1)
    return dequantized_frame


hot_pixel_index = torch.tensor(torch.load(
    str(Path() / 'hot_pixel_indices.pt')
)[11]).cuda()
offsets = torch.tensor(list(product([-2, 2], [-2, 2]))).cuda()

def resample_hot_pixels(mosaic_float):
    nbds = hot_pixel_index[..., None, :].add(offsets)
    resample_values = mosaic_float[..., nbds[..., -2], nbds[..., -1]].mean(dim=-1)
    mosaic_float[..., hot_pixel_index[..., -2], hot_pixel_index[..., -1]] = resample_values
    return mosaic_float


def mosaic_to_aligned_demosaic(mosaic):
    demosaic = F.pixel_unshuffle(mosaic, downscale_factor=2)
    upscaled_demosaic = F.interpolate(
        demosaic,
        size=tuple(it * 2 - 1 for it in demosaic.shape[-2:]),
        align_corners=True,
        mode='bilinear',
        antialias=False
    )
    output = torch.stack(
        [
            F.pad(upscaled_demosaic[..., 0, :, :], [0, 1, 0, 1], 'replicate'),
            F.pad(upscaled_demosaic[..., 1, :, :], [1, 0, 0, 1], 'replicate'),
            F.pad(upscaled_demosaic[..., 2, :, :], [0, 1, 1, 0], 'replicate'),
            F.pad(upscaled_demosaic[..., 3, :, :], [1, 0, 1, 0], 'replicate'),
        ], dim=-3
    )
    return output


def angles_to_stokes(angles):
    I_Q = (angles[..., 3, :, :] + angles[..., 0, :, :])
    I_U = (angles[..., 1, :, :] + angles[..., 2, :, :])
    stokesI = torch.stack([I_Q, I_U]).amax(0)
    stokesQ = (angles[..., 3, :, :] - angles[..., 0, :, :])
    stokesU = (angles[..., 1, :, :] - angles[..., 2, :, :])
    angles_dict = {
        '0': angles[..., 3, :, :],
        '45': angles[..., 1, :, :],
        '90': angles[..., 0, :, :],
        '135': angles[..., 2, :, :],
    }
    stokes_dict = {
        'I': stokesI,
        'Q': stokesQ,
        'U': stokesU,
        'L': torch.complex(stokesQ, stokesU),
    }
    return angles_dict, stokes_dict
