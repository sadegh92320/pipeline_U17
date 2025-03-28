import pickle
from datetime import datetime, timedelta
from lag_video import first_non_zero_speed
import pandas as pd
from get_scene_2 import get_video_darkening_times
import os
import datetime as dt

def fix_time(time_obj):
    if isinstance(time_obj, dt.time):
        corrected_seconds = time_obj.hour * 60 + time_obj.minute
        return corrected_seconds
    return None

output_folder = "data_eye"
s_to_check = []

csv_filename = "new_trim.csv"

four_scene = [6, 34, 42, 47, 50, 53, 100, 105, 113, 128, 131, 133, 147, 158, 159,
              161, 179, 183, 185, 219, 230, 253, 277, 340, 31, 32] 
with open("participants.pkl", "rb") as file:
    driving = pickle.load(file)

with open("participants_eye.pkl", "rb") as file:
    eyes = pickle.load(file)

with open("scene_2.pkl", "rb") as file:
    scene_2 = pickle.load(file)

df_1 = pd.read_excel("scene_1.xlsx", engine = "openpyxl")
df_1["Participant"] = df_1["Participant"].apply(lambda x: int(x) if pd.notna(pd.to_numeric(x, errors="coerce")) else x)
df_1["seconds"] = df_1["time of 1"].apply(fix_time)


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
            first_1 = 1
        if i == 1:
            scene = [3,5]
            first = 3
        if i == 2:
            scene = [4,8]
            first = 4
        eye = (participant[key][i])
        parse_nb = (eye)[0].split("_")

        if int(parse_nb[0]) < 239 or int(parse_nb[0]) == 336:
            continue
        
                
        
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
                
                #first_time = start_driving
                        
                #lag = first_non_zero_speed(eye[4], first_time, eye[3])
                
                break
            if int(f1[2][" SceneNr"][0]) == first_1: 
                
                
                row_interest = df_1.loc[(df_1['Participant']) == int(parse_nb[0]), "First scenario"].iloc[0]
                
                try:
                    if int(row_interest) == 0:
                        
                       
                        time_starting = df_1.loc[df_1["Participant"] == int(parse_nb[0]), "seconds"].iloc[0]
                        
                        
                        seconds_between_start = (start_driving - f1[3]).total_seconds()
                       
                        
                        total_seconds = time_starting + seconds_between_start
                       
                       
                        time_scene_1 = f1[3]
                      
                        event_time = eye[3] + timedelta(seconds=time_starting)
                        diff = event_time - time_scene_1
                        seconds_difference = diff.total_seconds()
                        lag = (seconds_difference, int(total_seconds))
                        break
                        
                       

                    elif int(row_interest) == 1:
                        time_starting = df_1.loc[df_1["Participant"] == int(parse_nb[0]), "seconds"].iloc[0]
                        
                        total_seconds = time_starting 
                       
                       
                        
                      
                        event_time = eye[3] + timedelta(seconds=time_starting)
                        diff = event_time - start_driving
                        seconds_difference = diff.total_seconds()
                        lag = (seconds_difference, int(total_seconds))
                        break
                    else:
                        break
                except Exception as e:
                   
                    print(e)     
                    break
            #    lag = get_video_darkening_times(eye[4], first_time, eye[3])
                        
            #    break
        if lag == 0 or lag[1] == None:
            passed.append(eye[0])
            with open("error_participant.txt", "a") as outfile:  # Open file in append mode
                    outfile.write(f"{eye[0]}\n")
            continue
        else:
            #Do - 
            beginning_eye = eye[3] - timedelta(seconds=lag[0])
            new_end_eye =end_eye - timedelta(seconds = lag[0])


        add_time = 0
        last_start = 0
        for f1 in (filtered_tuples):
          
            start_in = False
            end_in = False
            scenario = (f1[2][" SceneNr"][0])
            s_to_check.append(scenario)

            start_driving = datetime.strptime(f1[2]["UTC"][0], "%Y-%m-%d %H:%M:%S:%f")
            end_driving = datetime.strptime(f1[2]["UTC"].iloc[-2], "%Y-%m-%d %H:%M:%S:%f")
            if last_start != 0:
                add_time = add_time + (start_driving - last_start).total_seconds()
            
            start_vid = lag[1] + add_time
            end_vid = (end_driving - start_driving).total_seconds() + start_vid

            difference = abs(start_driving - new_end_eye)
            if start_driving > beginning_eye and start_driving < new_end_eye and difference > timedelta(seconds=15):
                start_in = True
        
            
            if end_driving > beginning_eye and end_driving < (new_end_eye + timedelta(seconds=15)):
                end_in = True
            i_1 = 0
            i_2 = 0
            for index, line in eye[2].iterrows():
                if line["timestamp"] > start_vid and i_1 == 0:
                    i_1 = index
                if line["timestamp"] > end_vid and i_2 ==0:
                    i_2 = index
                    break
            if i_2 == 0:
                i_2 = len(eye[2])
            new_df = eye[2][i_1:i_2]

            
            save_path = os.path.join(output_folder, f"{parse_nb[0]}_{scenario}_{start_vid}.pkl")
            
            new_df.to_pickle(save_path)

                
            data[(parse_nb[0], scenario)] = {"participant number": parse_nb[0],
                                             "Scenario number": scenario,
                                             "index 1": i_1,
                                             "index 2": i_2,
                                             "dataframe eye": str(save_path),
                                             
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
                                                       "video_path":str(eye[4]),

                                                       }
            
            new_row = pd.DataFrame([{
                        "participant number": parse_nb[0],
                        "Scenario number": scenario,
                        "index 1": i_1,
                        "index 2": i_2,
                        "dataframe eye": str(save_path),
                        
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
                        "video_path":str(eye[4]),
                    }])


            new_row.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)
            last_start = start_driving


                
df = pd.DataFrame.from_dict(data, orient='index')

# Reset index to remove multi-index and flatten it into columns
df.reset_index(drop=True, inplace=True)

# Save to CSV
csv_filename = "trimming.csv"
df.to_csv(csv_filename, index=False)

print(s_to_check)