from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
from train_number import NumberCNN
import torchvision.transforms as transforms


# Load model weights
model = NumberCNN()

model.load_state_dict(torch.load("number_cnn.pth"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

from PIL import Image

def detect_number_in_roi(roi):
    """Detects the number in the given region of interest (ROI) using the trained CNN model."""
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))  # Convert NumPy array to PIL image
    roi_tensor = transform(roi_pil).unsqueeze(0)  # Apply transformations and add batch dimension
    
    with torch.no_grad():
        output = model(roi_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    return predicted_label


def first_non_zero_speed(video_path, start_1, time_eye):
    """ Detects first occurrence of speed values 1 or 2 in the video using YOLO and CNN."""
    acceptable = {1}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Unable to retrieve FPS.")
        cap.release()
        return None, None

    timestep = 1 / fps
    yolo_model = YOLO("runs/detect/train2/weights/best.pt")
    frame_idx = 0
    seconds_difference = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, show=False, save=False)
        timestamp = frame_idx * timestep

        for result in results:
            for box in result.boxes:
                cls = box.cls[0]
                if int(cls.item()) == 0:  # Ensure it's the correct class
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    roi = frame[ymin:ymax, xmin:xmax]

                    detected_number = detect_number_in_roi(roi)

                    # Draw bounding box and label on frame
                    #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    #cv2.putText(frame, str(detected_number), (xmin, ymin - 10),
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    #cv2.imshow("Detected Frame", frame)
                    #key = cv2.waitKey(0)  # Wait for user to press a key before proceeding to the next frame
                    #if key == 27:  # Press 'Esc' to exit early
                    #    cap.release()
                    #    cv2.destroyAllWindows()
                    #    return None, None
                    
                    print("Detected Number:", detected_number)
                    print("Timestamp:", timestamp)
                    
                    if detected_number in acceptable:
                        print(f"Valid Detection: {detected_number} at {timestamp} seconds")
                        event_time = time_eye + timedelta(seconds=timestamp)
                        diff = event_time - start_1
                        seconds_difference = diff.total_seconds()
                        cap.release()
                        cv2.destroyAllWindows()
                        return seconds_difference, timestamp
                    break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return seconds_difference, None