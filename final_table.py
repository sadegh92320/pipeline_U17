
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

folder = r"\Users\Student\Desktop\driving_data_test"
participants = []



        
    
class table_participant:
    def __init__(self, folder, participant_info, folder_eye, video = None, pickel = None):
        self.f = folder
        self.data_eye = {}
        self.eye_data = folder_eye
        self.participants = self.get_participants()
        self.participants_eye = self.get_participants_gaze()
        self.data = {}
        self.pickel = pickel
        self.info = pd.read_excel(participant_info)
        self.df = self.get_df()
        
        self.eye_tracker = None
        self.video = video
    
    
    
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
               
                data_eye = convert(imu_path, gaze_path)
               
                
                if os.path.isfile(participant_path):
                  
                    with open(participant_path, 'r') as f:
                        data = json.load(f)    
                        number = data['name']
                participants_eye.append([number, data_eye, data_eye.loc[data_eye['label'] == "eye tracker"].reset_index(drop=True), time1, video])
        
        return participants_eye
            
                
    def get_particition_video(self):
        first_scene = [1, 3, 4]
       
        self.participants_eye
        self.participants    
        for eye in self.participants_eye:
            scene = []
            parse_nb = eye[0].split("_")
            
            filtered_tuples = [t for t in self.participants if int(t[0]) == int(parse_nb[0])]
          
            
            seen = set()
            for f1 in reversed(filtered_tuples):
                if int(f1[2][" SceneNr"][0]) == 0:
                    continue
                
                
                
                
               
                start_driving = datetime.strptime(f1[2]["UTC"][0], "%Y-%m-%d %H:%M:%S:%f")

                end_driving = datetime.strptime(f1[2]["UTC"].iloc[-1], "%Y-%m-%d %H:%M:%S:%f")
                
                seconds_to_add = (eye[2]["timestamp"].iloc[-1])
            
                
                
                end_eye = eye[3] + timedelta(seconds=seconds_to_add)
                

                
                
                if int(f1[2][" SceneNr"][0]) not in seen:
                   
                    seen.add(int(f1[2][" SceneNr"][0]))
                    if start_driving > eye[3] and start_driving < end_eye:
                        scene.append([int(f1[2][" SceneNr"][0]), start_driving, end_driving])
                        if int(f1[2][" SceneNr"][0]) in first_scene:
                            eye.append(start_driving)

            
            
                
            #reversed(scene)
            
            for s, start, end in scene:
                                                                                                                                                                       
                partition_1 = (start - eye[3]).total_seconds()
                partition_2 = (end - eye[3]).total_seconds()
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
                part_number_set = set()

                if ' PedSpawned' in file.columns and ' CarSpawned' in file.columns:
                    
                    condition = (file[' CarSpawned'] == " True") | (file[' PedSpawned'] == " True")
                    
                    true_indices = file.index[condition].tolist()
                if true_indices:  # Only add if condition is met
                    
                    part_number_set.add(part_number)
                
                
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


                
                last_non_zero = non_zero_indices[-1]

                
                trimmed_df = file.iloc[first_non_zero:last_non_zero].reset_index(drop=True)
                
                if not trimmed_df.empty:
                    participants.append((part_number, formatted_date, trimmed_df))
            except:
                
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
            sub_df[' Brake'] = pd.to_numeric(sub_df[' Brake'], errors='coerce').fillna(0)
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
        
        for participant in self.participants_eye:
            #num_fix = fixation_AOI(participant[1], participant[3], participant[4:])
            i = 0
            #{'side mirror': 12, 'reer mirror': 5, 'speed': 20}
            #{'side mirror': 0.25712266666666644, 'reer mirror': 0.3686891999999986, 'speed': 0.4989025500000011}
            if len(participant) < 6:
                continue
            
            parse_nb = participant[0].split("_")
            
            
            diff_time = abs(participant[5] - participant[3])
            if diff_time > timedelta(minutes = 1, seconds = 30):
                passed_participant.append(int(parse_nb[0]))
                continue
           

            lag = first_non_zero_speed(participant[4], participant[5], participant[3])
            
            for fix in participant[6:]:
                print((int(parse_nb[0]), fix[0]))
                if int(fix[0]) == 7 or int(fix[0]) == 6:
                    half = int((fix[1] + fix[2])/2)
                    fix_1 = fix[:-1] + [half]  
                    fix_2 = [fix[0]] + [half] + [fix[2]]
                    num_fix_1 = num_fix = fixation_AOI(participant[2], participant[4], fix_1, int(parse_nb[0]), fix[0], lag)
                    num_fix_2 = num_fix = fixation_AOI(participant[2], participant[4], fix_2, int(parse_nb[0]), fix[0], lag)
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
                    
                    num_fix = fixation_AOI(participant[2], participant[4], fix, int(parse_nb[0]), fix[0], lag)
                    
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
                        
                    
                
              
                    
                    
            df = pd.DataFrame(self.data_eye.values())
            df.to_csv('participant_result_eye.csv', index=False)
            print(passed_participant)

                
            #print(eye_data)
        

    def get_df(self):
        # Loop through your data and build the rows
       
    
        for nb, date, files in self.participants:
           
            if int(files[" SceneNr"][0]) != 0:
                
    
                
                
                key = (nb, int(files[" SceneNr"][0]))  # Unique key based on participant and scenario number
                if key in self.data and self.data[key]["Date"] > date:
                    pass
                else:
                    continue 
               
                
                age = self.info.loc[(self.info['Participant number ']) == int(nb), 'Age']
                gender = self.info.loc[(self.info['Participant number ']) == int(nb), 'Gender']
                event = self.info.loc[(self.info['Participant number ']) == int(nb), 'How many events have you attended']
               
                
               
                
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
                



                

                
                
                self.data[key] = {
                        'Date': date,
                        'Participant number': nb,
                        'Scenario number': int(files[" SceneNr"][0]),
                        'number of steering': self.get_steering(files),
                        'number of braking': self.get_brake(files),
                        'average velocity': self.get_velocity(files),
                        'steering velocity': self.get_steer_velocity(files),
                        'max steering angle': self.get_max_steer_angle(files),
                        'max decceleration': self.get_max_min_acceleration(files)[1],
                        'max acceleration': self.get_max_min_acceleration(files)[0],
                        'reaction time': self.get_reaction_time(files),
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
t = table_participant(folder, part_info, r"\Users\Student\Desktop\U17ccetg")


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
