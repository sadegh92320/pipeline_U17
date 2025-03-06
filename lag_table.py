import pickle
from datetime import datetime, timedelta
from lag_video import first_non_zero_speed
import pandas as pd
from get_scene_2 import get_video_darkening_times


four_scene = [6, 34, 42, 47, 50, 53, 100, 113, 128, 131, 133, 147, 158, 159,
              161, 179, 183, 185, 219, 230, 253, 277, 340] 
with open("driving.pkl", "rb") as file:
    driving = pickle.load(file)

with open("eye.pkl", "rb") as file:
    eyes = pickle.load(file)

list_eye = []
participant = {}
for e in eyes:
    parse_nb = e[0].split("_")
    nb = int(parse_nb[0])
    try:
        participant[nb].append(e)
    except:
        participant[nb] = [e]

for key in participant:
    participant[key].sort(key=lambda x: x[3]) 


normal = [1, 2, 7, 6]
passed = []
first = 0
data = {}
for key in (participant):
    for i in range(len(participant[key])):
        if i == 0:
            scene = [1,2,7,6]
            night = 2
        if i == 1:
            scene = [3,5]
            first = 3
        if i == 2:
            scene = [4,8]
            first = 4
        eye = (participant[key][i])
        parse_nb = (eye)[0].split("_")
                
                
        filtered_tuples = [t for t in driving if (int(t[0]) == int(parse_nb[0]) and int(t[2][" SceneNr"][0]) in scene)]

        
       
                    
            
                
        seen = set()
        lag = 0
        seconds_to_add = (eye[2]["timestamp"].iloc[-1])
       
                    
        end_eye = eye[3] + timedelta(seconds=seconds_to_add)
            
        for f1 in (filtered_tuples):
            lag = 0
            scenario = int(f1[2][" SceneNr"][0])
                
                    
                    
                
                    
            if int(f1[2][" SceneNr"][0]) == 0:
                continue      
                    
                
            start_driving = datetime.strptime(f1[2]["UTC"][0], "%Y-%m-%d %H:%M:%S:%f")

            end_driving = datetime.strptime(f1[2]["UTC"].iloc[-2], "%Y-%m-%d %H:%M:%S:%f")

            

            if int(f1[2][" SceneNr"][0]) == first: 
                
                first_time = start_driving
                        
                lag = first_non_zero_speed(eye[4], first_time, eye[3])
                
                break
            if int(f1[2][" SceneNr"][0]) == night: 
                
                first_time = start_driving
                        
                lag = get_video_darkening_times(eye[4], first_time, eye[3])
                        
                break
        if lag == 0 or lag[1] == None:
            passed.append(eye[0])
            continue
        else:
            beginning_eye = eye[3] - timedelta(seconds=lag[0])
            new_end_eye =end_eye - timedelta(seconds = lag[0])


                
        for f1 in reversed(filtered_tuples):
                
            start_in = False
            end_in = False
            scenario = (f1[2][" SceneNr"][0])
            start_driving = datetime.strptime(f1[2]["UTC"][0], "%Y-%m-%d %H:%M:%S:%f")
            end_driving = datetime.strptime(f1[2]["UTC"].iloc[-2], "%Y-%m-%d %H:%M:%S:%f")
            print(lag)
            start_vid = lag[1]
            end_vid = (end_driving - start_driving).total_seconds() + lag[1]

            difference = abs(start_driving - end_eye)
            if start_driving > beginning_eye and start_driving < new_end_eye and difference > timedelta(seconds=15):
                start_in = True
            if end_driving < (new_end_eye + timedelta(seconds=15)):
                end_in = True

                
            data[(parse_nb[0], scenario)] = {"participant number": parse_nb[0],
                                             "Scenario number": scenario,
                                            "start drive": start_driving, 
                                                       "end drive": end_driving,
                                                       "initial start eye": eye[3],
                                                       "initial end eye": end_eye,
                                                       "lag": lag[0],
                                                       "new start eye": beginning_eye,
                                                       "new end eye": new_end_eye,
                                                       "star video": start_vid,
                                                       "end video": end_vid,
                                                       "start in": start_in,
                                                       "end in": end_in,

                                                       }


                
df = pd.DataFrame.from_dict(data, orient='index')

# Reset index to remove multi-index and flatten it into columns
df.reset_index(drop=True, inplace=True)

# Save to CSV
csv_filename = "trimming.csv"
df.to_csv(csv_filename, index=False)