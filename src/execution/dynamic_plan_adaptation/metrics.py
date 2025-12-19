import os

import cv2
import numpy as np
import yaml

from src.utils import load_cache_with_lock, save_cache_with_lock


def rgb_video_quality_assessment(video_data_path):
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    use_cache = config["execution_settings"]["use_cache"]
    cache_path = config["execution_settings"]["assessment_cache_path"]
    
    if use_cache:
        cache = load_cache_with_lock(cache_path)
        if video_data_path in cache["rgb-video-quality-assessment"]:
            return cache["rgb-video-quality-assessment"][video_data_path]
    
    cap = cv2.VideoCapture(video_data_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_data_path}")
        return "Cannot open video file."

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_sum = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.astype(np.float32) / 255.0

        brightness = (
            0.2126 * frame[:, :, 2] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 0]
        )
        brightness_sum += np.mean(brightness)
        frame_count += 1

    cap.release()

    if frame_count > 0:
        avg_brightness = brightness_sum / frame_count
    else:
        return "No frames found in video."

    if use_cache:
        cache["rgb-video-quality-assessment"][video_data_path] = avg_brightness
        save_cache_with_lock(cache_path, cache, "rgb-video-quality-assessment", video_data_path)
    
    return avg_brightness

def depth_video_quality_assessment(video_data_path):
    config_path = os.environ.get("CONFIG_PATH")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    use_cache = config["execution_settings"]["use_cache"]
    cache_path = config["execution_settings"]["assessment_cache_path"]
    
    if use_cache:
        cache = load_cache_with_lock(cache_path)
        if video_data_path in cache["depth-video-quality-assessment"]:
            return cache["depth-video-quality-assessment"][video_data_path]
    
    cap = cv2.VideoCapture(video_data_path)

    noise_means = []
    noise_stddevs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        noise_residual = frame - blurred_frame
        if np.any(noise_residual):
            noise_mean = np.mean(noise_residual)
            noise_stddev = np.std(noise_residual)

            noise_means.append(noise_mean)
            noise_stddevs.append(noise_stddev)

    cap.release()

    if noise_means and noise_stddevs:
        avg_noise_stddev = np.mean(noise_stddevs)
        if use_cache:
            cache["depth-video-quality-assessment"][video_data_path] = avg_noise_stddev
            save_cache_with_lock(cache_path, cache, "depth-video-quality-assessment", video_data_path)
        
        return avg_noise_stddev
    else:
        if use_cache:
            cache["depth-video-quality-assessment"][video_data_path] = None
            save_cache_with_lock(cache_path, cache, "depth-video-quality-assessment", video_data_path)
        
        return None


map_from_function_name_to_function = {
    "rgb-video-quality-assessment": rgb_video_quality_assessment,
    "depth-video-quality-assessment": depth_video_quality_assessment,
}

