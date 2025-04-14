import numpy as np
import time
import psutil
import torch
import multiprocessing as mp
import pynvml
import cv2
from supervision.geometry.dataclasses import Point, Vector


def update_activation_line(bbox: np.ndarray, keypoints: np.ndarray, ball_y: int):
    """ Dynamically update the activation line based on the ball's position. """
    leftKnee, rightKnee = keypoints[13], keypoints[14]
    leftHip, rightHip = keypoints[11], keypoints[12]
    nose = keypoints[0]
    upper_bound = bbox[1]
    knee_y = min(leftKnee[1], rightKnee[1])
    hip_y = min(leftHip[1], rightHip[1])

    if ball_y < upper_bound:
        JPart = "Head"
        activation_line = Vector(start=Point(float(bbox[0]), float(upper_bound)),
                                 end=Point(float(bbox[2]), float(upper_bound)))
    elif hip_y >= ball_y > nose[1]:
        JPart = "Torso"
        activation_line = Vector(start=Point(float(bbox[0]), float(hip_y)),
                                 end=Point(float(bbox[2]), float(hip_y)))
    elif knee_y >= ball_y > hip_y:
        JPart = "Thigh"
        activation_line = Vector(start=Point(float(bbox[0]), float(knee_y)),
                                 end=Point(float(bbox[2]), float(knee_y)))
    else:
        JPart = "Foot"
        activation_line = Vector(start=Point(float(bbox[0]), float(knee_y)),
                                 end=Point(float(bbox[2]), float(knee_y)))

    return activation_line, JPart


class JuggleTracker:
    def __init__(self):
        self.juggle_count = 0
        self.consistent_juggles = 0  # Tracks streak of successful juggles
        self.last_juggle_time = None  # Time of the last juggle
        self.first_juggle_time = None  # Time when the first juggle happened
        self.juggle_timestamps = []  # List to store time of each juggle
        self.ball_heights = []  # Store ball height at each juggle
        self.ball_x_positions = []  # Store ball X position for horizontal consistency
        self.ball_drops = 0

    def register_juggle(self, ball_x, ball_y):
        """ Tracks juggling count, consistency, and time gaps. """
        current_time = time.time()

        if self.first_juggle_time is None:
            self.first_juggle_time = current_time

        time_gap = 0
        if self.last_juggle_time is not None:
            time_gap = current_time - self.last_juggle_time

        self.last_juggle_time = current_time
        self.juggle_count += 1
        self.consistent_juggles += 1
        self.juggle_timestamps.append(current_time)

        # ✅ Store X and Y values for consistency calculation
        self.ball_x_positions.append(ball_x)
        self.ball_heights.append(ball_y)

        return self.juggle_count, time_gap

    def register_drop(self):
            """ ✅ New: Track when the ball drops """
            self.ball_drops += 1  # ✅ Increase drop count
            print(f"⚠️ Ball Dropped! Total Drops: {self.ball_drops}")

    def calculate_consistency(self, weight_abs=0.6, weight_rel=0.4, size_adjustment_k=5, epsilon=1e-3):
        """ 
        Evaluates consistency based on X-axis (horizontal) and Y-axis (vertical) positions.
        Ensures normalization and prevents excessive penalization.
        """

        if len(self.ball_x_positions) > 1 and len(self.ball_heights) > 1:
            x_positions = np.array(self.ball_x_positions)
            y_positions = np.array(self.ball_heights)

            median_x = np.median(x_positions)
            median_y = np.median(y_positions)

            mad_x = np.median(np.abs(x_positions - median_x)) + epsilon
            mad_y = np.median(np.abs(y_positions - median_y)) + epsilon

            range_x = np.max(x_positions) - np.min(x_positions) + epsilon
            range_y = np.max(y_positions) - np.min(y_positions) + epsilon

            # ✅ Normalize MAD values by range
            normalized_dev_x = mad_x / range_x
            normalized_dev_y = mad_y / range_y

            # ✅ Log transform to reduce extreme penalties
            total_deviation = (
                weight_abs * np.log1p(mad_x**2 + mad_y**2) +
                weight_rel * np.log1p(normalized_dev_x**2 + normalized_dev_y**2)
            )

            # ✅ Adjust scaling factor to prevent over-penalizing small datasets
            N = len(x_positions)
            scaling_factor = (N / (N + size_adjustment_k)) ** 0.5

            # ✅ Normalize the consistency score to [0, 1]
            consistency_score = max(
            0, 1.0 - scaling_factor * total_deviation / 10)  # Scale deviation

            # ✅ Debugging Output
            print(
            f"[DEBUG] Normalized MAD X: {normalized_dev_x:.4f}, Normalized MAD Y: {normalized_dev_y:.4f}")
            print(f"[DEBUG] Log Scaled Total Deviation: {total_deviation:.4f}")
            print(f"[DEBUG] Final Consistency Score: {consistency_score:.4f}")

            return round(consistency_score, 2)

        return 1.0  # Default to full consistency before enough data

    def calculate_time_gap(self):
        """ Returns the average time gap between juggles. """
        if len(self.juggle_timestamps) > 1:
            # Calculate time differences
            gaps = np.diff(self.juggle_timestamps)
            return np.mean(gaps)  # Return average time gap
        return 0

    def get_total_time(self):
        """ Returns the total time taken from first to last juggle. """
        if self.first_juggle_time is None or self.last_juggle_time is None:
            return 0  # No valid juggles yet
        return self.last_juggle_time - self.first_juggle_time

    def reset_consistency(self):
        """ Resets the consistency streak when the ball drops. """
        self.consistent_juggles = 0


def process_activation_line(data_queue, result_queue):
    """ Multiprocessing function to update activation lines. """
    while True:
        data = data_queue.get()
        if data is None:
            break
        bbox, keypoints, ball_y = data

        activation_line, JPart = update_activation_line(
            bbox, keypoints, ball_y)
        result_queue.put((activation_line, JPart))


def start_multiprocessing():
    """Starts multiprocessing for activation line updates."""
    data_queue = mp.Queue()
    result_queue = mp.Queue()
    process = mp.Process(target=process_activation_line,
                         args=(data_queue, result_queue))
    process.start()
    return data_queue, result_queue, process


def track_performance(prev_time, frames_since_last_check):
    pynvml.nvmlInit()  # Initialize NVML
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    """Track GPU, CPU usage, FPS, and latency."""
    current_time = time.perf_counter()
    latency = (current_time - prev_time) * 1000  # Convert to ms
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    fps = frames_since_last_check / \
        (current_time - prev_time + 1e-6)  # Fix FPS calculation

    print(f"CPU Usage: {cpu_usage:.2f}% | Memory Usage: {memory_usage:.2f}% | "
          f"Latency: {latency:.2f} ms | FPS: {fps:.2f} |" f"GPU Usage: {utilization.gpu}% | Memory Used: {memory.used / (1024**2):.2f} MB")

    return current_time  # Return updated timestamp
