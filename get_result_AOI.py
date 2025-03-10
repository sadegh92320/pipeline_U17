import pandas as pd
import pickle
from eye_traking_analysis import fixation_AOI
import math
import os



def saccade_amplitude(self, f):
        sacc = False
        point = 0
        distances = []
        
        for index, line in f.iterrows():
           
            if line["eye_movement"] == "saccade" and sacc == False:
                point = (float(line["gaze2d_x"]), float(line["gaze2d_y"]))
                sacc = True
            if line["eye_movement"] == "fixation" and sacc == True:
                
                distance = math.sqrt((float(line["gaze2d_x"]) - point[0])**2 + (line["gaze2d_y"] - point[1])**2)
                
                if not math.isnan(distance):
                    distances.append(distance)
                sacc = False
        
        
        avg = sum(distances)/len(distances)
        
       
       
        return avg

def saccade_velocity(self, f):
        sacc = False
        point = 0
        velocities = []
        
        for index, line in f.iterrows():
           
            if line["eye_movement"] == "saccade" and sacc == False:
                point = (float(line["gaze2d_x"]), float(line["gaze2d_y"]))
                time = float(line["timestamp"])
                sacc = True
            if line["eye_movement"] == "fixation" and sacc == True:
                
                distance = math.sqrt((float(line["gaze2d_x"]) - point[0])**2 + (line["gaze2d_y"] - point[1])**2)
                velocity = distance/(float(line["timestamp"] - time))
                
                if not math.isnan(velocity):
                    velocities.append(velocity)
                sacc = False
        
        
        avg = sum(velocities)/len(velocities)
        
       
       
        return avg

def duration_fixation(self, f):
        fix = False
        time = 0
        times = []
        
        for index, line in f.iterrows():
           
            if line["eye_movement"] == "fixation" and fix == False:
                time = line["timestamp"]
                fix = True
            if line["eye_movement"] == "saccade" and fix == True:
                times.append(float(line["timestamp"]) - float(time))
                fix = False
        
        avg = sum(times)/len(times)
       
        return avg


csv_file = "new_trim.csv"
csv_filename = "result_gaze_data.cs"

with open('eye.pkl', 'rb') as file:
    data = pickle.load(file)

df = pd.read_csv(csv_file)

for index, row in df.iterrows():
    start = row["star video"]
    end = row["end video"]
    num_fix = fixation_AOI(row["dataframe eye"], row["video_path"], [start, end], int(row["participant number"]), int(row["Scenario number"]))
                    
                   
    amplitude = saccade_amplitude(row["dataframe eye"])
    velocity = saccade_velocity(row["dataframe eye"])
    duration = duration_fixation(row["dataframe eye"])
    new_row = pd.DataFrame([{
                            'Participant number': int(row["participant number"]),
                            'Scenario number': int(row["Scenario number"]),
                            'Saccade amplitude': amplitude,
                            'Saccade velocity': velocity,
                            'Duration fixation': duration,
                            'fixation in side mirror': num_fix[0]["side mirror"],
                            'fixation in rear mirror': num_fix[0]["reer mirror"],
                            'fixation in speed': num_fix[0]["speed"],
                            'duration fixation in side mirror': num_fix[1]["side mirror"],
                            'duration fixation in rear mirror': num_fix[1]["reer mirror"],
                            'duration fixation in speed': num_fix[1]["speed"],
                            'number speed in view': num_fix[2]["speed"],
                            'number rear mirror in view': num_fix[2]["reer mirror"],
                            'number side mirror in view': num_fix[2]["side mirror"],
    
                        }])
    
    new_row.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)