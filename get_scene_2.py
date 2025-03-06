import cv2
import numpy as np
from datetime import datetime, timedelta

#start_1, time_eye,

def get_video_darkening_times(video_path, start_1, time_eye, threshold=5):
    """
    Analyzes a video and finds timestamps where the overall brightness significantly decreases.
    
    Parameters:
        video_path (str): Path to the video file.
        threshold (float): Minimum percentage decrease in brightness to consider as "getting darker".
    
    Returns:
        List of timestamps (in seconds) where the brightness drops significantly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    brightness_values = []
    timestamps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale to measure brightness
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_frame)

        brightness_values.append(avg_brightness)
        timestamps.append(len(brightness_values) / frame_rate)

    cap.release()

    # Find times where brightness drops significantly
    darkening_times = []
    for i in range(1, len(brightness_values)):
        change = (brightness_values[i] - brightness_values[i - 1]) / brightness_values[i - 1] * 100
        if change < -threshold and timestamps[i] > 60:  # Significant decrease
            timestamp = timestamps[i]
            print(timestamp)
            
            event_time = time_eye + timedelta(seconds=timestamp)
            diff = event_time - start_1
            seconds_difference = diff.total_seconds()
            return seconds_difference, timestamp
           
    return None

# Example usage

#get_video_darkening_times("/Users/sadeghemami/eye_tracking_participant/20230815T095017Z/scenevideo.mp4")