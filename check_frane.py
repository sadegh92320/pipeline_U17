import pandas as pd
import os
import cv2

df = pd.read_csv("new_trim.csv")
output_folder = "frame_5s"
os.makedirs(output_folder, exist_ok=True)
not_in = []

for index, row in df.iterrows():
    if row["start in"] == True and row["end in"] == True:
      



        for index, row in df.iterrows():
            video_path = row["video_path"]
            start_time = row["star video"]  # Start time in seconds
            end_time = row["end video"]      # End time in seconds
            participant_number = row["participant number"]
            scenario_number = row["Scenario number"]

            # Construct unique filename using start_time
            save_path = os.path.join(output_folder, f"{participant_number}_{scenario_number}_{start_time}_frame.png")

            # Open video
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Cannot open video {video_path}")
                continue

            # Get FPS and total duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps  # Total video length in seconds

            # Compute target time (5s after start)
            frame_time = start_time + 5

            # Ensure frame_time does not exceed video duration or end_time
            if frame_time > duration or frame_time > end_time:
                print(f"Skipping {video_path}: 5s mark is beyond video end time.")
                cap.release()
                continue

            # Convert to frame index
            frame_index = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read multiple frames (some codecs need extra reads)
            for _ in range(3):
                ret, frame = cap.read()
                if ret:
                    break

            cap.release()

            if not ret:
                print(f"Error: Could not retrieve frame at {frame_time}s for {video_path}")
                continue

            # Save frame (overwrite if it already exists)
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")

    else:
        not_in.append((row["participant number"], row["Scenario number"]))

print(not_in)
print(len(not_in))
