from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])

def preprocess_green_digits(roi):
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

   
    lower_green = np.array([40, 50, 50])  
    upper_green = np.array([80, 255, 255])  

   
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

   
    result = cv2.bitwise_and(roi, roi, mask=cleaned_mask)
    grayscale_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return grayscale_result

def read_text_from_roi(roi):
   
    preprocessed_roi = preprocess_green_digits(roi)
    result = reader.readtext(preprocessed_roi, detail=0) 
    if result:
        try:
            
            text = ''.join(filter(str.isdigit, result[0]))
            return int(text) if text else None
        except ValueError:
            return None
    return None



def first_non_zero_speed(video_path, start_1, time_eye):
    
    #time_difference = start_time - end_time

   
    #threshold = timedelta(minutes=1, seconds=30)
    #if time_difference <= threshold:
    #    print("The time difference is within 1 minute 30 seconds.")
    #else:
    #    print("The time difference exceeds 1 minute 30 seconds.")
    acceptable = [1, 2, 3, 4, 5, 6, 7]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Unable to retrieve FPS.")
        return
    timestep = 1 / fps

    model = YOLO("runs/detect/train2/weights/best.pt") 

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        results = model(frame, show=False, save=False)
        timestamp = (frame_idx * timestep)

        for result in results:
            for box in result.boxes:
                cls = box.cls[0]
                if int(cls.item()) == 0:  
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                    
                    roi = frame[ymin:ymax, xmin:xmax]

                    
                    detected_number = read_text_from_roi(roi)
                    print(detected_number)
                    if detected_number is not None:
                        
                        if int(detected_number) in acceptable:
                            print("time")
                            print(timestamp)
                            
                            time = time_eye + timedelta(seconds=timestamp)
                            diff = time - start_1
                            seconds_difference = diff.total_seconds()

                           
                            cap.release()

        frame_idx += 1

    cap.release()
    print(seconds_difference)
    return seconds_difference


video_path = r"\Users\Student\Desktop\U17ccetg\20240716T081849Z\scenevideo.mp4"
