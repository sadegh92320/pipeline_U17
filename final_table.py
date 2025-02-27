
import pandas as pd
import os
from math import sqrt
from datetime import datetime, timedelta
import csv
import json
import math
from convert_gaze import convert
from eye_traking_analysis import fixation_AOI
import pytz
from lag_video import first_non_zero_speed
from get_position_driving_data import add_spawn
import numpy as np
from get_scene_2 import get_video_darkening_times
import traceback

#folder = r"\Users\Student\Desktop\driving_data_test"
#J'ai fait le 12 Avril
folder = "/Users/sadeghemami/21Feb2023"
participants = []



        
    
class table_participant:
    def __init__(self, folder, participant_info, folder_eye, pickel = None):
        self.f = folder
        self.data_eye = {}
        self.eye_data = folder_eye
        self.scene_2 = {}
        self.participants = self.get_participants()
        self.participants_eye = self.get_participants_gaze()
        self.data = {}
        self.pickel = pickel
        self.info = pd.read_excel(participant_info)
        self.df = None #self.get_df()
        
        
        self.eye_tracker = None
       
    
    
    
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
    
    def get_AOI_fixation(self):
        fixation_AOI(self.eye_data, self.video)
    

    
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

    
    def get_participants_gaze(self):
        participants_eye = []
        for root, dirs, files in os.walk(self.eye_data):
            for dir in dirs:
                # Build the correct path to meta/participant
                if dir == "meta":
                    continue
                
                time1 = datetime.strptime(dir, "%Y%m%dT%H%M%SZ")
                
                gmt = pytz.timezone("GMT")
                uk_time = pytz.timezone("Europe/London")  # Automatically handles DST

                # Localize the time to GMT
                time1_gmt = gmt.localize(time1)
                time1_uk = time1_gmt.astimezone(uk_time)


                # Convert to UK time
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
                try:
                    data_eye = convert(imu_path, gaze_path)
                except:
                    print(participant_path)
                    continue
                eye_tracker_data = data_eye.loc[data_eye['label'] == "eye tracker"].reset_index(drop=True)
                
                if int(eye_tracker_data["timestamp"][0]) > 1:
                    eye_tracker_data["timestamp"] = eye_tracker_data["timestamp"] - eye_tracker_data["timestamp"][0]
              
               
                
                if os.path.isfile(participant_path):
                  
                    with open(participant_path, 'r') as f:
                        data = json.load(f)    
                        number = data['name']
                participants_eye.append([number, data_eye, eye_tracker_data, time1, video])
        
        return participants_eye
            
                
    def get_particition_video(self):
        first_scene = [3, 4]
       
        
        for eye in self.participants_eye[:]:
            scene = []
            parse_nb = eye[0].split("_")
            
            
            filtered_tuples = [t for t in self.participants if int(t[0]) == int(parse_nb[0])]
            
                  
          
            
            seen = set()
            print(eye[0])
            lag = 0
            
           
            for f1 in (filtered_tuples):
                print("jjeke")
                print(len(filtered_tuples))
                print("scene")
                print(int(f1[2][" SceneNr"][0]))
                
               
                
                if int(f1[2][" SceneNr"][0]) == 0:
                    continue      
                
               
                start_driving = datetime.strptime(f1[2]["UTC"][0], "%Y-%m-%d %H:%M:%S:%f")
                print(start_driving)

                end_driving = datetime.strptime(f1[2]["UTC"].iloc[-2], "%Y-%m-%d %H:%M:%S:%f")
                
                seconds_to_add = (eye[2]["timestamp"].iloc[-1])
                
                end_eye = eye[3] + timedelta(seconds=seconds_to_add)
        

                
                if start_driving > eye[3] and start_driving < end_eye:
                    if int(f1[2][" SceneNr"][0]) == 2:
                        first_time = datetime.strptime(self.scene_2[int(parse_nb[0])][0], "%Y-%m-%d %H:%M:%S:%f")
                        lag = get_video_darkening_times(eye[4], first_time, eye[3])
                        eye.append(lag)
                        beginning_eye = eye[3] - timedelta(seconds=lag[0])
                        end_eye =end_eye - timedelta(seconds = lag[0])
                        break

                    if int(f1[2][" SceneNr"][0]) in first_scene:
                        print("laagag")
                        print(int(f1[2][" SceneNr"][0]))
                    
                        first_time = start_driving
                        
                        lag = first_non_zero_speed(eye[4], first_time, eye[3])
                        print(first_time)
                        print(eye[3])
                        print(lag)
                        eye.append(lag)
                        beginning_eye = eye[3] - timedelta(seconds=lag[0])
                        print(beginning_eye)
                        end_eye =end_eye - timedelta(seconds = lag[0])
                        print(end_eye)
                        
                        break
            if lag == 0:
                with open("error_participant.txt", "a") as outfile:  # Open file in append mode
                    outfile.write(f"{eye[0]}\n")
                self.participants_eye.remove(eye)
            for f1 in reversed(filtered_tuples):
               
                
                if int(f1[2][" SceneNr"][0]) == 0:
                    continue      
                
               
                start_driving = datetime.strptime(f1[2]["UTC"][0], "%Y-%m-%d %H:%M:%S:%f")

                end_driving = datetime.strptime(f1[2]["UTC"].iloc[-2], "%Y-%m-%d %H:%M:%S:%f")   
                if int(f1[2][" SceneNr"][0]) not in seen:
                   
                    seen.add(int(f1[2][" SceneNr"][0]))
                    print(int(f1[2][" SceneNr"][0]))
                    print(start_driving)
                    difference = abs(start_driving - end_eye)

                    if start_driving > beginning_eye and start_driving < end_eye and difference > timedelta(seconds=15):
                        print(int(f1[2][" SceneNr"][0]))
                        scene.append([int(f1[2][" SceneNr"][0]), start_driving, end_driving])

            
            print("sssss")
            print(len(scene))
            print(scene)
            
            
          
            
            for s, start, end in scene:
                
                                                                                                                                                                       
                partition_1 = (start - beginning_eye).total_seconds()
                partition_2 = (end - beginning_eye).total_seconds()
                partition_2 = partition_2 
                print((start-end).total_seconds())
                print("partition")
                print(partition_1)
                print(partition_2)
                
                i_1 = 0
                i_2 = 0 

                for index, line in eye[2].iterrows():
                    
                    if line["timestamp"] > partition_1 and i_1 == 0:
                        i_1 = index
                       
                    if line["timestamp"] > partition_2 and i_2 == 0:
                        
                        i_2 = index
                       

                if i_2 == 0:
                    i_2 = len(eye[2])
               
                    
                eye.append([s, i_1, i_2])
                
                        
                
        
                    
                    
    def get_participants(self):
        participants = []
        for filename in sorted(os.listdir(self.f)):
            
            f = os.path.join(self.f, filename)
          
            if os.path.isfile(f) and os.path.getsize(f) >= 50 * 1024:  # 50 KB in bytes
                # Process the file
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
                        self.scene_2[part_number].append(file["UTC"][0])
                    except:

                        self.scene_2[part_number] = [file["UTC"][0]]
                part_number_set = set()
                spawn_indice = 0
            

                if ' PedSpawned' in file.columns and ' CarSpawned' in file.columns:
                    
                    condition = (file[' CarSpawned'] == " True") | (file[' PedSpawned'] == " True")
                    
                    true_indices = file.index[condition].tolist()
                    if len(true_indices) >0:
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
        print(len(part_number_set))
        return participants
    
    
    def get_steering(self, f):
        g = []
        for i in range(3,len(f)-1):
            try:
                gradient = (float(f[" Steering"][i])-float(f[" Steering"][i-3]))/((float(f[" Time"][i]) - float(f[" Time"][i-3])))
                g.append(gradient)
            except:
                continue
        change = 0
        for i in range(1, len(g) - 1):
            if ((g[i-1] > 0 and g[i] < 0) or (g[i-1] < 0 and g[i] > 0)):
                change = change + 1 
        return change
    
    def get_brake(self, f):
        g = []
        for i in range(3,len(f)-1):
            try:
                gradient = (float(f[" Brake"][i])-float(f[" Brake"][i-3]))/((float(f[" Time"][i]) - float(f[" Time"][i-3])))
                g.append(gradient)
            except:
                continue
        change = 0
        for i in range(1, len(g) - 1):
            if ((g[i-1] > 0 and g[i] < 0) or (g[i-1] < 0 and g[i] > 0)):
                change = change + 1 
        return change
    
    def get_velocity(self, f):
        total = 0
        nb = 0
        for i in range(1,len(f)-1):
            try:
                velocity = sqrt((float(f[" Velocity x"][i])*float(f[" Velocity x"][i])) + (float(f[" Velocity z"][i])*float(f[" Velocity z"][i])))
                if velocity > 16 or velocity < 4:
                    continue
                else:
                    total += velocity
                    nb += 1
            except:
                continue
        try:
            return (total/nb)
        except:
            return 0
    
    def get_steer_velocity(self, f):
        total = 0
        nb = 0
        
        for i in range(3,len(f)-1):
            try:
                gradient = abs(((float(f[" Steering"][i]) - float(f[" Steering"][i-3]))*90) / (float(f[" Time"][i]) - float(f[" Time"][i-3])))
                total += gradient
                nb += 1
            except:
                continue
        try:
            return total/nb
        except:
            return 0
    
    def get_max_steer_angle(self, f):
        series = pd.to_numeric(f[" Steering"], errors='coerce').dropna()
        series = series.astype(float)
        return series.max()
    
    def get_max_min_acceleration(self, f):
        acc = []
        
        for i in range(3,len(f)-1):

            try:
                velocity1 = sqrt((float(f[" Velocity x"][i])*float(f[" Velocity x"][i])) + (float(f[" Velocity z"][i])*float(f[" Velocity z"][i])))
                velocity2 = sqrt((float(f[" Velocity x"][i-3])*float(f[" Velocity x"][i-3])) + (float(f[" Velocity z"][i-3])*float(f[" Velocity z"][i-3])))
                if velocity1 > 16 or velocity2 > 16:
                    continue
                gradient = ((velocity1 - velocity2) / (float(f[" Time"][i]) - float(f[" Time"][i-3])))
                acc.append(gradient)
            except:
                
                continue
        return (max(acc),min(acc))
    
    def get_accident_speed(self, f):
        try:
            speed = f.loc[(f[' CollidedWithTarget']).str.strip() == "True"]
            if len(speed) > 1:
                line = speed.iloc[0]
                velocity = sqrt((float(line[" Velocity x"])*float(line[" Velocity x"])) + (float(line[" Velocity z"])*float(line[" Velocity z"])))
                return velocity
            else:
                return "NA"
        except:
            return "No collide with target for this test"


    
    def get_reaction_time(self, f):
        #print(f[' PedSpawned'])
        pedscenario = [5,6,7,8]
        if int(f[" SceneNr"][0]) not in pedscenario:  
                    return "NA"
        
        t_ped = f.loc[(f[' PedSpawned']).str.strip() == "True", ' Time']
        
        if len(t_ped) > 1:
            t_collision = f.loc[(f[' CollidedWithTarget']).str.strip() == "True", ' Time']
            t_ped = t_ped.iloc[0]
            start_i = f[f[' PedSpawned'].str.strip() == "True"].index[0]
            sub_df = f.loc[start_i:]
            sub_df.loc[:, ' Brake'] = pd.to_numeric(sub_df[' Brake'], errors='coerce').fillna(0)

            t_brake = sub_df.loc[sub_df[' Brake'] > 0, " Time"]     
            if len(t_brake) > 1:
                if len(t_collision) > 0 and t_collision.iloc[0] > t_brake.iloc[0]:
                    return -1
                return float(t_brake.iloc[0]) - float(t_ped)
            else: 
                return -1
        else:
            return 0
            
            
    def diff_steer(self, part_nb, steer):
        steer_1 = self.data[(part_nb,1)]["number of steering"]
       
        return ((steer - steer_1)/steer_1)*100
    
    def diff_brake(self, part_nb, brake):
        brake_1 = self.data[(part_nb,1)]["number of braking"]
        try:
            return ((brake - brake_1)/brake_1)*100
        except:
            return 100
    
    def diff_avg_vel(self, part_nb, avg_vel):
        avg_vel_1 = self.data[(part_nb,1)]["average velocity"]
        return ((avg_vel - avg_vel_1)/avg_vel_1)*100
    
    def diff_steer_vel(self, part_nb, steer_vel):
        steer_vel_1 = self.data[(part_nb,1)]["steering velocity"]
        return ((steer_vel - steer_vel_1)/steer_vel_1)*100
    
    def diff_steer_max(self, part_nb, max_steer):
        max_steer_1 = self.data[(part_nb,1)]["max steering angle"]
        return ((max_steer - max_steer_1)/max_steer_1)*100
    
    def diff_decc(self, part_nb, max_decc):
        max_decc_1 = self.data[(part_nb,1)]["max decceleration"]
        return ((max_decc - max_decc_1)/max_decc_1)*100
    
    def diff_acc(self, part_nb, max_acc):
        max_acc_1 = self.data[(part_nb,1)]["max acceleration"]
        return ((max_acc - max_acc_1)/max_acc_1)*100


    def get_df_eye(self):
        passed_participant = []
        df = pd.DataFrame(self.data_eye.values())
        
        for participant in self.participants_eye:
            
            #num_fix = fixation_AOI(participant[1], participant[3], participant[4:])
            i = 0
            #{'side mirror': 12, 'reer mirror': 5, 'speed': 20}
            #{'side mirror': 0.25712266666666644, 'reer mirror': 0.3686891999999986, 'speed': 0.4989025500000011}
            if len(participant) < 6:
                continue
            
            parse_nb = participant[0].split("_")
            
            
            #diff_time = abs(participant[5] - participant[3])
            #if diff_time > timedelta(minutes = 1, seconds = 30):
                #print("passed as the ")
               
                #passed_participant.append(int(parse_nb[0]))
                #continue
           

            #lag = first_non_zero_speed(participant[4], participant[5], participant[3])
            new_data = []
            for fix in participant[6:]:
                print(int(fix[0]))
                print(int(parse_nb[0]))
                
                
                
               
                if int(fix[0]) == 7 or int(fix[0]) == 6:
                   
                    half = int((fix[1] + fix[2])/2)
                    fix_1 = fix[:-1] + [half]  
                    fix_2 = [fix[0]] + [half] + [fix[2]]
                    num_fix_1 =  fixation_AOI(participant[2], participant[4], fix_1, int(parse_nb[0]), fix[0], participant[5])
                    num_fix_2 =  fixation_AOI(participant[2], participant[4], fix_2, int(parse_nb[0]), fix[0], participant[5], 2)
                    key = (int(parse_nb[0]), fix[0])
                    amplitude = self.saccade_amplitude(participant[2][fix[1]:fix[2]])
                    velocity = self.saccade_velocity(participant[2][fix[1]:fix[2]])
                    duration = self.duration_fixation(participant[2][fix[1]:fix[2]])
                    self.data_eye[key] = {
                            'Participant number': int(parse_nb[0]),
                            'Scenario number': fix[0],
                            'Saccade amplitude': amplitude,
                            'Saccade velocity': velocity,
                            'Duration fixation': duration,
                            'fixation in side mirror': num_fix_1[0]["side mirror"] + num_fix_2[0]["side mirror"],
                            'fixation in rear mirror': num_fix_1[0]["reer mirror"] + num_fix_2[0]["side mirror"],
                            'fixation in speed': num_fix_1[0]["speed"] + num_fix_2[0]["side mirror"],
                            'duration fixation in side mirror': num_fix_1[1]["side mirror"] + num_fix_2[0]["side mirror"],
                            'duration fixation in rear mirror': num_fix_1[1]["reer mirror"] + num_fix_2[0]["side mirror"],
                            'duration fixation in speed': num_fix_1[1]["speed"] + num_fix_2[0]["side mirror"],
                            'number speed in view': num_fix_1[2]["speed"] + num_fix_2[0]["side mirror"],
                            'number rear mirror in view': num_fix_1[2]["reer mirror"] + num_fix_2[0]["side mirror"],
                            'number side mirror in view': num_fix_1[2]["side mirror"] + num_fix_2[0]["side mirror"],

                        }
                  
               
            
                else:
                    
                    
                    num_fix = fixation_AOI(participant[2], participant[4], fix, int(parse_nb[0]), fix[0], participant[5][1])
                    
                    key = (int(parse_nb[0]), fix[0])
                    print(fix[1])
                    print(fix[2])
                    amplitude = self.saccade_amplitude(participant[2][fix[1]:fix[2]])
                    velocity = self.saccade_velocity(participant[2][fix[1]:fix[2]])
                    duration = self.duration_fixation(participant[2][fix[1]:fix[2]])
                    self.data_eye[key] = {
                            'Participant number': int(parse_nb[0]),
                            'Scenario number': fix[0],
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

                        }
                new_data.append(self.data_eye[key])

        
            df = pd.DataFrame(new_data)

            df.to_csv('participant_result_eye.csv', mode='a', index=False, header=not os.path.isfile('participant_result_eye.csv'))
                        

    def get_df(self):
        # Loop through your data and build the rows
       
        
        for nb, date, files in self.participants:
            
           
            if int(files[" SceneNr"][0]) != 0:           
                key = (nb, int(files[" SceneNr"][0]))  # Unique key based on participant and scenario number
               
                if key in self.data and self.data[key]["Date"] > date:
                    continue
                
               
                
                age = self.info.loc[(self.info['Participant number ']) == int(nb), 'Age']
                gender = self.info.loc[(self.info['Participant number ']) == int(nb), 'Gender']
                event = self.info.loc[(self.info['Participant number ']) == int(nb), 'How many events have you attended']
                if " CollidedWithBystander" in files.columns:
                    number_of_collision = files[" CollidedWithBystander"].iloc[-2]
                else:
                    
                    number_of_collision = "NA"
                has_collided = files[" CollidedWithTarget"].iloc[-2]
                #files['total_velocity'] = np.sqrt(files[' Velocity x']**2 + files[' Velocity y']**2 + files[' Velocity z']**2)

                # Compute standard deviation of total velocity
                #std_dev = files['total_velocity'].std()
               
                
               
                
                if int(nb) == 15 or int(nb) == 56:
                    continue
                if int(event) >= 10:
                    regular = "Yes"
                else:
                    regular = "No"
                license_driving = self.info.loc[(self.info['Participant number ']) == int(nb), 'Do you have a driving license?']
              
                age = int(age)
                self.get_accident_speed(files)

                diff_steer = 0
                diff_brake = 0
                diff_avg_vel = 0
                diff_steer_vel = 0
                diff_steer_max = 0
                diff_decc_max = 0
                diff_acc_max = 0

                if int(files[" SceneNr"][0]) in [1,5,6,7,8]:
                    diff_steer = "NA"
                    diff_brake = "NA"
                    diff_avg_vel = "NA"
                    diff_steer_vel = "NA"
                    diff_steer_max = "NA"
                    diff_decc_max = "NA"
                    diff_acc_max = "NA"
                else:
                   
                    diff_steer = self.diff_steer(nb, self.get_steering(files))
                    diff_brake = self.diff_brake(nb, self.get_brake(files))
                    diff_avg_vel = self.diff_avg_vel(nb, self.get_velocity(files))
                    diff_steer_vel = self.diff_steer_vel(nb, self.get_steer_velocity(files))
                    diff_steer_max = self.diff_steer_max(nb, self.get_max_steer_angle(files))
                    diff_decc_max = self.diff_decc(nb, self.get_max_min_acceleration(files)[1])
                    diff_acc_max = self.diff_acc(nb, self.get_max_min_acceleration(files)[0])
                



                

                
                if (gender.tolist())[0] != "Male" or (gender.tolist())[0] != "Female":
                    print((gender.tolist())[0])
                
                self.data[key] = {
                        'Date': date,
                        'Participant number': nb,
                        'Scenario number': int(files[" SceneNr"][0]),
                        'number of steering': self.get_steering(files),
                        'number of braking': self.get_brake(files),
                        'average velocity': self.get_velocity(files),
                        #'standard deviation velocity': std_dev,
                        'steering velocity': self.get_steer_velocity(files),
                        'max steering angle': self.get_max_steer_angle(files),
                        'max decceleration': self.get_max_min_acceleration(files)[1],
                        'max acceleration': self.get_max_min_acceleration(files)[0],
                        'reaction time': self.get_reaction_time(files),
                        'Collided during scenario': has_collided,
                        'Number of Collision': number_of_collision,
                        'speed of accident': self.get_accident_speed(files),
                        'Age': age,
                        'Gender': (gender.tolist())[0],
                        'Is regular to the charity': regular,
                        'Has a driving license': (license_driving.tolist())[0],
                        'number of steering difference (%)': diff_steer,
                        'number of braking difference (%)': diff_brake,
                        'average velocity difference (%)': diff_avg_vel,
                        'steering velocity difference (%)': diff_steer_vel,
                        'max steering difference (%)': diff_steer_max,
                        'max deccelration difference (%)': diff_decc_max,
                        'max acceleration difference (%)': diff_acc_max
                    }
            
        
        if self.pickel != None:
            with open(self.pickel, 'rb') as file:
                df = pd.read_pickle(file)

            
            new_df = pd.DataFrame(self.data.values())

            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(self.data.values())
       
        
        return df
    def generate_pickle(self):

        self.df.to_pickle("participant_result.pkl")
    
    def generate_csv(self):
        self.df.to_csv('participant_result.csv', index=False)


#folder = "Scenarios_Charity/test"
part_info = "participant_info_3.xlsx"
#print(convert("../eye_tracking_participant/20241030T091636Z/imudata.gz", "../eye_tracking_participant/20241030T091636Z/gazedata.gz"))
#pkl_file = "participant_result.pkl"
t = table_participant(folder, part_info, "/Users/sadeghemami/eye_tracking_participant")
#pickel="participant_result.pkl"
#r"\Users\Student\Desktop\U17ccetg"

#print(t.eye_data)
(t.get_particition_video())
#print(t.participants_eye)
t.get_df_eye()
#(t.get_particition_video())
#print(t.data_eye)
#t.saccade_velocity()
#print(t.df.shape)

#t.generate_csv() 
#t.generate_pickle()
#t.generate_csv()
