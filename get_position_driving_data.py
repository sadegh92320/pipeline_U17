import os 
import pandas as pd 

#if pos z > 801 for 1,2,3,4 True

#if pos x > 514 for 5 True

#if pos x > 460 for 6 True

#if pos x > 553 and pos z > 435 7 True

#if pos z > 555 8 True

def add_spawn(df):
    scene_nb = int(df[" SceneNr"][0])
    df[' Time'] = df[' Time'] - df[' Time'].iloc[0]  # Subtract the first time value

    # Calculate time difference (delta_time)
    df['delta_time'] = df[' Time'].diff().fillna(0)  # First value is zero

    # Calculate position based on velocity and delta_time
    df['pos x'] = (df[' Velocity x'] * df['delta_time']).cumsum()
    df['pos y'] = (df[' Velocity y'] * df['delta_time']).cumsum()
    df['pos z'] = (df[' Velocity z'] * df['delta_time']).cumsum()
    spawn_index = 0

    if scene_nb in [1,2,3,4]:
        spawn_index = df[df['pos z'] >= 801].index.min() + 86
        if pd.notnull(spawn_index):
            df.loc[spawn_index:, ' CarSpawned'] = True
    if scene_nb == 5:
        spawn_index = df[df['pos x'] >= 514].index.min() + 86
        if pd.notnull(spawn_index):
            df.loc[spawn_index:, ' PedSpawned'] = True
    if scene_nb == 6:
        spawn_index = df[df['pos x'] >= 460].index.min() + 86
        if pd.notnull(spawn_index):
            df.loc[spawn_index:, ' PedSpawned'] = True
    if scene_nb == 7:
        spawn_index = df[(df['pos z'] > 553) & (df['pos x'] > 435)].index.min() + 86
        if pd.notnull(spawn_index):
            df.loc[spawn_index:, ' PedSpawned'] = True

    if scene_nb == 8:
        spawn_index = df[df['pos z'] >= 555].index.min() + 86
        if pd.notnull(spawn_index):
            df.loc[spawn_index:, ' PedSpawned'] = True

    return spawn_index


            

                       
                
                
