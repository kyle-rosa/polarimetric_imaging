import os
import time
from pathlib import Path

import cv2
import torch

import processing
import streaming
import visualisation


class CameraStreamer():
    def __init__(
        self,
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
        resample_hot_pixels: bool,
        diffusion_time: float,
        gamma: float,
        long_exposure_frames : int,
        compute_device: torch.device,
        # Output settings:
        output_dir: (str | os.PathLike),
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.resample_hot_pixels = resample_hot_pixels
        self.gamma = gamma
        self.long_exposure_frames = long_exposure_frames
        self.cyclic_colour_map = visualisation.load_cyclic_colourmap(
            compute_device
        )
        self.camera_device = streaming.run_camera(
            frame_rate=frame_rate,
            exposure_time=exposure_time,
            width=pixels_width,
            height=pixels_height,
            bit_depth=bit_depth,
            throughput_reserve=throughput_reserve,
        )
        self.frame_generator = streaming.get_image_buffers(
            self.camera_device, compute_device
        )
        _ = self.frame_generator.send(None)
        self.output_manager = streaming.make_display_manager(
            frame_rate, pixels_width, pixels_height, 
            window_width, window_height, output_dir
        )
        self.output_manager.send(None)
        self.long_exposure_buffer = torch.zeros(
            (1, 1, pixels_height, pixels_width), 
            dtype=torch.float,
            device=compute_device
        )
        self.diffusion_multiplier = processing.make_diffusion_multiplier(
            width=pixels_width, 
            height=pixels_height, 
            time=diffusion_time
        ).to(compute_device)

    def make_long_exposure(self,):
        self.long_exposure_buffer.fill_(0)
        for _ in range(self.long_exposure_frames):
            mosaic = self.frame_generator.send(None)
            mosaic_float = processing.dequantize_frame(mosaic)
            if self.resample_hot_pixels:
                mosaic_float = processing.resample_hot_pixels(mosaic_float)
            self.long_exposure_buffer.add_(mosaic_float)
        long_exposure_buffer = self.long_exposure_buffer.div(self.long_exposure_frames)

        angles_tensor = processing.mosaic_to_aligned_demosaic(long_exposure_buffer)
        (angles_dict, stokes_dict) = processing.angles_to_stokes(angles_tensor)
        
        visualisations = visualisation.visualise_polarisation(
            angles_dict, stokes_dict, 
            self.cyclic_colour_map, 
            self.gamma, 
            self.diffusion_multiplier
        )

        timestamp = time.time()
        save_dir = self.output_dir / f'{str(timestamp)}' 
        for key in visualisations:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(
                str(save_dir / f'{key}.png'), visualisations[key][..., [2, 1, 0]]
            )

    def start_stream(self):
        print("""Press 'e' to to take a long exposure, or 'q' to exit.""")
        while True:
            match chr(cv2.waitKey(1) & 0xFF):
                case 'q':
                    break
                case 'e':
                    self.make_long_exposure()

            mosaic = self.frame_generator.send(None)

            mosaic_float = processing.dequantize_frame(mosaic)
            if self.resample_hot_pixels:
                mosaic_float = processing.resample_hot_pixels(mosaic_float)
            angles_tensor = processing.mosaic_to_aligned_demosaic(mosaic_float)
            (angles_dict, stokes_dict) = processing.angles_to_stokes(angles_tensor)
            visualisations = visualisation.visualise_polarisation(
                angles_dict, stokes_dict, 
                self.cyclic_colour_map, 
                self.gamma, 
                self.diffusion_multiplier
            )

            self.output_manager.send(visualisations)


if __name__ == '__main__':
    CameraStreamer(
        # Capture settings:
        frame_rate=10.0,
        exposure_time=95_000.0,
        pixels_width=2448,
        pixels_height=2048,
        bit_depth=12,
        # Data transfer settings:
        throughput_reserve=10,
        # Display settings:
        window_width=2448,
        window_height=2048,
        # Image processing settings:
        resample_hot_pixels=True,
        diffusion_time=32,
        gamma=(1/2.33),
        long_exposure_frames=64,
        compute_device=torch.device('cuda'),
        # Output settings:
        output_dir=Path()/'data/output',
    ).start_stream()
    