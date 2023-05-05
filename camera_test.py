import os
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

import processing
import streaming
import visualisation

import pandas as pd


def visualise_test_results(test_frames):
    bin_counts = torch.bincount(test_frames.reshape(-1)).div(test_frames.size(0))
    cumulative_pixels = bin_counts.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    df = pd.DataFrame(
        data=[it.cpu().numpy() for it in [bin_counts, cumulative_pixels]],
        index=['counts', 'cumulative']
    ).T
    df.plot(logy=True).get_figure().savefig(str(output_dir / 'bin_counts.png'))

    hot_pixel_img_dir = output_dir / 'hot_pixels'
    os.makedirs(hot_pixel_img_dir, exist_ok=True)
    
    frames_float = test_frames.float().mean(dim=0)[0]

    hot_pixel_dict = {}
    for threshold in range(0, 256, 1):
        hot_pixel_dict[threshold + 1] = frames_float.gt(threshold).nonzero().tolist()

    # print(hot_pixel_dict)
    torch.save(hot_pixel_dict, 'hot_pixel_indices.pt')


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

    for _ in tqdm(range(10_000)):
        mosaic = frame_generator.send(None)
        mosaic_float = processing.dequantize_frame(mosaic)
        angles_tensor = processing.mosaic_to_aligned_demosaic(mosaic_float)
        (angles_dict, stokes_dict) = processing.angles_to_stokes(angles_tensor)
        visualisations = visualisation.visualise_polarisation(
            angles_dict, stokes_dict, cyclic_colour_map, gamma
        )
        videos_writer.send(visualisations)

        key = (cv2.waitKey(1) & 0xFF)
        if key == ord("q"):
            break

        byte_depth_constants = torch.tensor(2).pow(
            torch.arange((1 if (bit_depth==8) else 2)).multiply(8)
        ).to(compute_device)

        num_test_frames = 32
        if key == ord("e"):
            # Begin Test
            timestamp = time.time()
            save_dir = output_dir / 'test_frames' 
            os.makedirs(save_dir, exist_ok=True)
            test_frames = []
            for _ in range(num_test_frames):
                mosaic = frame_generator.send(None)
                mosaic_long = mosaic.mul(byte_depth_constants).sum(dim=-1).subtract(1)
                test_frames.append(mosaic_long)
            test_frames = torch.cat(test_frames)
            torch.save(test_frames, save_dir / str(timestamp))
            visualise_test_results(test_frames)
            
            


            


if __name__ == '__main__':
    print(f'PyTorch {torch.__version__} | CUDA Toolkit {torch.version.cuda}')

    # Camera settings:
    frame_rate = 10.0
    exposure_time = 990_000.0 / frame_rate
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
    output_dir = Path() / "data/tests"
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
