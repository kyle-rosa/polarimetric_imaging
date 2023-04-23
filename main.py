import os
import time
from pathlib import Path

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
    os.makedirs(output_dir, exist_ok=True)
    cyclic_colour_map = visualisation.load_cyclic_colourmap(compute_device)
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

    videos_writer = streaming.make_display_manager(
        frame_rate, buffer_width, buffer_height, 
        window_width, window_height, output_dir
    )
    videos_writer.send(None)

    while True:
        mosaic = frame_generator.send(None)
        mosaic_float = processing.dequantize_frame(mosaic)
        angles_tensor = processing.mosaic_to_aligned_demosaic(mosaic_float)
        (angles_dict, stokes_dict) = processing.angles_to_stokes(angles_tensor)
        visualisations = visualisation.visualise_polarisation(angles_dict, stokes_dict, cyclic_colour_map, gamma)
        videos_writer.send(visualisations)

        key = (cv2.waitKey(1) & 0xFF)
        if key == ord("q"):
            break
        if key == ord("e"):
            timestamp = time.time()
            long_exposure_buffer = torch.zeros(
                (1, 1, buffer_height, buffer_width), 
                dtype=torch.float,
                device=compute_device
            )
            long_exposure_frames = 256
            for _ in range(long_exposure_frames):
                mosaic = frame_generator.send(None)
                mosaic_float = processing.dequantize_frame(mosaic)
                long_exposure_buffer.add_(mosaic_float)
            long_exposure_buffer.div_(long_exposure_frames)

            angles_tensor = processing.mosaic_to_aligned_demosaic(long_exposure_buffer)
            (angles_dict, stokes_dict) = processing.angles_to_stokes(angles_tensor)
            visualisations = visualisation.visualise_polarisation(angles_dict, stokes_dict, cyclic_colour_map, gamma)

            save_dir = output_dir / f'{str(timestamp)}' 
            for key in visualisations:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(str(save_dir / f'{key}.png'), visualisations[key][..., [2, 1, 0]])


if __name__ == '__main__':
    print(f'Running PyTorch {torch.__version__} with CUDA Toolkit {torch.version.cuda}...')

    # Camera settings:
    frame_rate = 10.0
    exposure_time = (990_000.0 / frame_rate)
    pixels_width = 2448
    pixels_height = 2048
    bit_depth = 12
    throughput_reserve = 10

    # Display settings:
    window_width = (2_448 // 2)
    window_height = (2_048 // 2)

    # Image processing settings:
    diffusion_time = 0.
    gamma = (1 / 2.33)
    compute_device = 'cuda'

    # Output settings:
    output_dir = (Path() / f"data/output")
    save_video = False
    save_frames = False


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
        save_video=save_video,
        save_frames=save_frames
    )
