import os
from pathlib import Path

import time
import cv2
import torch
from tqdm import tqdm

import processing
import streaming
import visualisation


def display_video_stream(
    # Camera settings:
    frame_rate: float,
    exposure_time: float,
    pixels_width: int,
    pixels_height: int,
    bit_depth: int,
    throughput_reserve: int,
    # Display settings:
    window_width: int,
    window_height: int,
    # Image processing settings:
    diffusion_time: float,
    gamma: float,
    compute_device: torch.device,
    # Output settings:
    output_dir: (str | os.PathLike),
    save_video=True,
    save_frames=False
):
    camera_device = streaming.run_camera(
        frame_rate=frame_rate,
        exposure_time=exposure_time,
        width=pixels_width,
        height=pixels_height,
        bit_depth=bit_depth,
        throughput_reserve=throughput_reserve,
    )
    key = -1
    frame_generator = streaming.get_image_buffers(camera_device, compute_device)
    mosaic = frame_generator.send(None)
    (_, _, buffer_height, buffer_width, bit_depth) = mosaic.shape

    cyclic_colour_map = visualisation.load_cyclic_colourmap(compute_device)

    os.makedirs(output_dir, exist_ok=True)
    videos_writer = streaming.make_display_manager(
        frame_rate, 
        buffer_width, buffer_height, 
        window_width, window_height,
        output_dir
    )
    videos_writer.send(None)
    
    for step in tqdm(range(10_000)):
        mosaic = frame_generator.send(None)

        mosaic_float = processing.dequantize_frame(mosaic)
        angles_tensor = processing.mosaic_to_aligned_demosaic(mosaic_float)
        (angles_dict, stokes_dict) = processing.angles_to_stokes(angles_tensor)

        visualisations = visualisation.visualise_polarisation(angles_dict, stokes_dict, cyclic_colour_map, gamma)
        videos_writer.send(visualisations)
        key = (cv2.waitKey(1) & 0xFF)
        if key == ord("q"):
            break


if __name__ == "__main__":
    # Camera settings:
    frame_rate = 10.0
    exposure_time = 95_000.0
    pixels_width = 2448
    pixels_height = 1960
    bit_depth = 12
    throughput_reserve = 10

    # Display settings:
    window_width = (2_048 // 2)
    window_height = (2_048 // 2)

    # Image processing settings:
    diffusion_time = 0.
    gamma = (1 / 2.33)
    compute_device = 'cuda'

    # Output settings:
    output_dir = (Path() / f"data/output/{str(time.time())}")
    save_video=False
    save_frames=False


    display_video_stream(
        frame_rate=frame_rate,
        exposure_time=exposure_time,
        pixels_width=pixels_width,
        pixels_height=pixels_height,
        bit_depth=bit_depth,
        throughput_reserve=throughput_reserve,
        window_width=window_width,
        window_height=window_height,
        diffusion_time=diffusion_time,
        gamma=gamma,
        compute_device=compute_device,
        output_dir=output_dir,
        save_video=False,
        save_frames=False
    )
