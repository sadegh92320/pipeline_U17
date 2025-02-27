import pickle
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

def get_participants(folder, participants_pkl="participants.pkl", scene_2_pkl="scene_2.pkl"):
    participants = []
    scene_2 = {}

    for filename in sorted(os.listdir(folder)):
        f = os.path.join(folder, filename)

        if os.path.isfile(f) and os.path.getsize(f) >= 50 * 1024:  # 50 KB in bytes
            pass
        else:
            continue

        moving = filename.split("_")     
        part_number = moving[0]
        date = moving[2].replace(".csv", "")
        formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%d-%m-%Y")
        
        try:
            file = pd.read_csv(f)
            
            if int(file[" SceneNr"][0]) == 2:
                try: 
                    scene_2[int(part_number)].append(file["UTC"][0])
                except:
                    scene_2[int(part_number)] = [file["UTC"][0]]

            spawn_indice = 0

            if ' PedSpawned' in file.columns and ' CarSpawned' in file.columns:
                condition = (file[' CarSpawned'] == " True") | (file[' PedSpawned'] == " True")
                true_indices = file.index[condition].tolist()
                
                if len(true_indices) > 0:
                    spawn_indice = true_indices[0] + 110
                else:
                    try:
                        spawn_indice = add_spawn(file)
                    except:
                        print(part_number)
                        print(file[" SceneNr"][0])
                        traceback.print_exc()
            else:
                file[' PedSpawned'] = "False"
                file[' CarSpawned'] = "False"
                spawn_indice = add_spawn(file)

            collision = 0
            if ' CollidedWithTarget' in file.columns:
                condition = (file[' CollidedWithTarget'] == " True")
                true_indices = file.index[condition].tolist()
                if len(true_indices) > 0:
                    collision = true_indices[0] + 53
            else:
                file[' CollidedWithTarget'] = "False"

            file[' Throttle'] = pd.to_numeric(file[' Throttle'], errors='coerce').fillna(0)
            throttle_threshold = 0.08
            consistency_check_window = 5

            non_zero_indices = file[file[' Throttle'] != 0].index
            non_zero_2 = file[file[' Throttle'] > 0.1].index

            for idx in non_zero_2:
                if idx + consistency_check_window - 1 < len(file):  # Ensure we don't exceed the data length
                    if all(file[' Throttle'][idx:idx + consistency_check_window] > throttle_threshold):
                        first_non_zero = idx
                        break  

            try:
                last_non_zero = non_zero_indices[-1]
            except:
                print(part_number)
                print(file[" SceneNr"][0])

            if collision != 0:
                trimmed_df = file.iloc[first_non_zero:collision].reset_index(drop=True)
            elif spawn_indice != 0 and not np.isnan(spawn_indice):
                trimmed_df = file.iloc[first_non_zero:spawn_indice].reset_index(drop=True)
            else:
                trimmed_df = file.iloc[first_non_zero:last_non_zero].reset_index(drop=True)

            if not trimmed_df.empty:
                participants.append((part_number, formatted_date, trimmed_df))

        except Exception as e:
            traceback.print_exc()
            print(e)
            continue

    # **Save participants as a pickle file**
    with open(participants_pkl, 'wb') as f:
        pickle.dump(participants, f)
    
    # **Save scene_2 as a separate pickle file**
    with open(scene_2_pkl, 'wb') as f:
        pickle.dump(scene_2, f)

    print(f"Pickle files saved: {participants_pkl}, {scene_2_pkl}")

    return participants
