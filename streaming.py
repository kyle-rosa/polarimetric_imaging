import os

import arena_api.system
import cv2
import numpy as np
import torch
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def select_device_from_user_input(
    selected_index: (int | None) = None
):
    device_infos = arena_api.system.system.device_infos
    print('Camera device information:\n', pd.DataFrame(device_infos))
    while selected_index is None:
        device_infos = arena_api.system.system.device_infos
        if len(device_infos) == 0:
            print("No camera connected\nPress enter to search again")
            input()
            continue
        print("Devices found:")
        for i in range(len(device_infos)):
            print(f"\t{i}. {device_infos[i]['model']} SN: {device_infos[i]['serial']}")
        while True:
            line = input("Selection: ")
            try:
                selected_index = int(line)
                if 0 <= selected_index < len(device_infos):
                    break
                else:
                    print(f"Please enter a valid number between 0 and {len(device_infos)-1}\n")
            except Exception:
                print("\nPlease enter a valid number\n")
    selected_model = device_infos[selected_index]['model']
    print(f"\nCreate device: {selected_model}...")
    device = arena_api.system.system.create_device(device_infos=device_infos[selected_index])[0]
    return device


def run_camera(
    frame_rate: (float | None) = None,
    exposure_time: (float | None) = None,
    width: int = 2448,
    height: int = 1960,
    bit_depth: int = 8,
    throughput_reserve: int = 10,
):
    device = select_device_from_user_input(selected_index=0)
    device.tl_stream_nodemap.get_node('StreamBufferHandlingMode').value = 'NewestOnly'
    device.tl_stream_nodemap.get_node('StreamPacketResendEnable').value = True
    device.tl_stream_nodemap.get_node('StreamAutoNegotiatePacketSize').value = True

    device.nodemap.get_node('PixelFormat').value = f"PolarizeMono{bit_depth}"
    device.nodemap.get_node('DeviceLinkThroughputReserve').value = throughput_reserve

    device.nodemap.get_node('OffsetX').value = 0
    device.nodemap.get_node('OffsetY').value = 0
    device.nodemap.get_node('Width').value = width
    device.nodemap.get_node('Height').value = height
    device.nodemap.get_node('OffsetX').value = (2448 - width) // 2
    device.nodemap.get_node('OffsetY').value = (2048 - height) // 2

    if frame_rate is not None:
        device.nodemap.get_node('AcquisitionFrameRateEnable').value = True
        device.nodemap.get_node('AcquisitionFrameRate').value = frame_rate
    else:
        device.nodemap.get_node('AcquisitionFrameRateEnable').value = False

    if exposure_time is not None:
        device.nodemap.get_node('ExposureAuto').value = 'Off'
        device.nodemap.get_node('GainAuto').value = 'Off'
        device.nodemap.get_node('ExposureTime').value = exposure_time
    else:
        device.nodemap.get_node('ExposureAuto').value = 'Continuous'
        device.nodemap.get_node('GainAuto').value = 'Off'

    return device


def auto_adjust_exposure_time(camera_device, demosaic_float):
    current_exposure_time = camera_device.nodemap.get_node('ExposureTime').value
    frame_rate = camera_device.nodemap.get_node('AcquisitionFrameRate').value
    max_pixel_value = demosaic_float.max()
    print(f'Maximum pixel value: {max_pixel_value}.')
    if max_pixel_value > 0.95:
        new_exposure_time = max(300.0, current_exposure_time * 0.95)
    else:
        new_exposure_time = min((1_000_000.0 / frame_rate * 0.95), current_exposure_time * 1.05)
    camera_device.nodemap.get_node('ExposureTime').value = new_exposure_time
    print(f"New exposure time: {new_exposure_time}.\n")
    return new_exposure_time


def get_image_buffers(
    camera_device: arena_api._device.Device,
    compute_device: torch.device
):
    camera_device.start_stream()
    while True:
        buffer = camera_device.get_buffer()
        buffer_shape = (buffer.height, buffer.width, buffer.bits_per_pixel // 8)
        frame_array = np.ctypeslib.as_array(buffer.pdata, shape=buffer_shape)
        frame_tensor = torch.from_numpy(frame_array)[None, None].to(device=compute_device)
        camera_device.requeue_buffer(buffer)
        yield frame_tensor



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


def make_display_manager(
    frame_rate: float, 
    buffer_width: int,
    buffer_height: int,
    window_width: int, 
    window_height: int,
    output_dir: (str | os.PathLike)
) -> None:
    """
    Coroutine that creates a window for displaying and saving videos. 
    It receives a stream of images as input, which it displays on the window and writes to disk. 
    
    Args:
        window_width (int): The width of the display window.
        window_height (int): The height of the display window.
    
    Yields:
        None
    """
    windows = {'L', 'ADoLP'}
    video_writers = {}
    for window in windows:
        cv2.startWindowThread()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, window_height, window_width)
        # cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for writer in video_writers:
        video_writers[writer] = cv2.VideoWriter(
            filename=str(output_dir / f'{writer}.mp4'), 
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
            fps=frame_rate,
            frameSize=(buffer_width, buffer_height)
        )
    while True:
        tensor_images_dict = yield
        for k in tensor_images_dict:
            images = to_numpy_byte_image(tensor_images_dict[k][0].expand(-1, -1, 3))
            if k in windows:
                cv2.imshow(k, images[..., [2, 1, 0]])
            if k in video_writers:
                video_writers[k].write(images[..., [2, 1, 0]])
