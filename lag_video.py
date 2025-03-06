from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timedelta
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

# Load ResNet-18 Model
def load_resnet18():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for grayscale
    model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Output layer for digits 0-9
    return model

# Load trained ResNet-18 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_resnet18().to(device)
model.load_state_dict(torch.load("resnet_digits.pth", map_location=device))
model.eval()

# Updated Preprocessing Function (Using CLAHE, Brightness Mask, and Green Mask)
def preprocess_image(roi, brightness_threshold=120):
    """Applies CLAHE for contrast enhancement, brightness filtering, and an improved green mask for digit extraction."""
    
    # Convert to grayscale
    image_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Extract the Green Channel
    green_channel = roi[:, :, 1]

    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image_gray)
    enhanced_green = clahe.apply(green_channel)

    # Compute adaptive green intensity thresholds
    green_min, green_max = np.percentile(enhanced_green, [5, 95])
    green_mask = cv2.inRange(enhanced_green, green_min, green_max)

    # Invert Green Mask so green areas appear white (255)
    corrected_green_mask = cv2.bitwise_not(green_mask)

    # Apply Morphological Closing on Green Mask
    kernel = np.ones((3,3), np.uint8)
    corrected_green_mask = cv2.morphologyEx(corrected_green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply Brightness Filtering
    bright_mask = (contrast_enhanced > brightness_threshold).astype(np.uint8) * 255

    # Combine the Corrected Green Mask with Brightness Mask
    final_mask = cv2.bitwise_and(bright_mask, corrected_green_mask)

    # Resize for ResNet (224x224)
    resized = cv2.resize(final_mask, (224, 224))

    # Convert to PyTorch tensor with Augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
    ])

    return transform(resized).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Function to detect number in ROI using ResNet-18
def detect_number_in_roi(roi):
    """Detects the number in the given region of interest (ROI) using the trained ResNet-18 model."""
    roi_tensor = preprocess_image(roi)
    
    with torch.no_grad():
        output = model(roi_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    return predicted_label

# Function to detect number in video
def first_non_zero_speed(video_path, start_1, time_eye, confidence_threshold=0.8):
    """Detects first occurrence of speed values in video using YOLO and ResNet-18 and visualizes detection."""
    
    acceptable = {1, 2, 3, 4, 5, 6, 7, 9}  # Target numbers
    in_row_number = []
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

        # YOLO detection
        results = yolo_model(frame, conf=confidence_threshold)  # Filter by confidence
        timestamp = frame_idx * timestep

        for result in results:
            for box in result.boxes:
                cls = box.cls[0]
                conf = box.conf[0].item()  # Confidence score

                if int(cls.item()) == 0 and conf >= confidence_threshold:  # Ensure correct class with confidence
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                    roi = frame[ymin:ymax, xmin:xmax]

                    detected_number = detect_number_in_roi(roi)

                    # Draw bounding box around detected AOI
                    #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Overlay detected number and confidence score
                    #label = f"Num: {detected_number}, Conf: {conf:.2f}"
                    #cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Show the frame with AOI highlighted
                    #cv2.imshow("Detection", frame)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
                    #    cap.release()
                    #    cv2.destroyAllWindows()
                    #    return None, None

                    print(f"Detected Number: {detected_number} at {timestamp:.2f}s (Confidence: {conf:.2f})")

                    if detected_number in acceptable:
                        in_row_number.append(detected_number)
                    else:
                        in_row_number = []
                    if len(in_row_number) == 10:
                        print(f"Valid Detection: {detected_number} at {timestamp:.2f} seconds")
                        event_time = time_eye + timedelta(seconds=timestamp)
                        diff = event_time - start_1
                        seconds_difference = diff.total_seconds()
                        cap.release()
                        cv2.destroyAllWindows()
                        return seconds_difference, timestamp

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return seconds_difference, None
