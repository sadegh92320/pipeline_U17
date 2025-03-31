import pickle
import os
import json
import pytz
from datetime import datetime
from convert_gaze import convert

path_etg = "C:/Users/imucl/Documents/U17CC/data/ETG"

def get_participants_gaze(eye_data, save_path="participants_eye.pkl", save_passed = "passed_participant.pkl"):
    passed = []
    participants_eye = []
    
    for root, dirs, files in os.walk(eye_data):
        for dir in dirs:
            if dir == "meta":
                continue
            
            try:

                time1 = datetime.strptime(dir, "%Y%m%dT%H%M%SZ")
                
                
                gmt = pytz.timezone("GMT")
                uk_time = pytz.timezone("Europe/London")  # Automatically handles DST
                
                # Localize the time to GMT and convert to UK time
                time1_gmt = gmt.localize(time1)
                time1_uk = time1_gmt.astimezone(uk_time)
                
                # Convert to UK time (preserving original format)
                time1 = datetime(
                    year=time1_uk.year,
                    month=time1_uk.month,
                    day=time1_uk.day,
                    hour=time1_uk.hour,
                    minute=time1_uk.minute,
                    second=time1_uk.second
                )

                participant_path = os.path.join(root, dir, "meta", "participant")
                imu_path = os.path.join(root, dir, "imudata.gz")
                gaze_path = os.path.join(root, dir, "gazedata.gz")
                video = os.path.join(root, dir, "scenevideo.mp4")
                if os.path.isfile(participant_path):
                    with open(participant_path, 'r') as f:
                        data = json.load(f)
                        number = data['name']
                print(number)
               

                try:
                  
                    data_eye = convert(imu_path, gaze_path)
                except Exception as e:
                    part_nb = number.split("_")
                    passed.append(part_nb[0])
                    print(f"Error processing {participant_path}: {e}")
                    continue

                eye_tracker_data = data_eye.loc[data_eye['label'] == "eye tracker"].reset_index(drop=True)
                
                if int(eye_tracker_data["timestamp"][0]) > 1:
                    eye_tracker_data["timestamp"] = eye_tracker_data["timestamp"] - eye_tracker_data["timestamp"][0]

                
                
                participants_eye.append([number, data_eye, eye_tracker_data, time1, video])
            
            except Exception as e:
                print(f"Skipping {dir} due to error: {e}")
    
    print(passed)
    print(len(passed))
    with open(save_path, 'wb') as f:
        pickle.dump(participants_eye, f)

    with open(save_passed, 'wb') as f:
        pickle.dump(passed, f)
        print("saved")
    
    print(f"Pickle file saved at: {save_path}")
    return participants_eye


get_participants_gaze(path_etg)