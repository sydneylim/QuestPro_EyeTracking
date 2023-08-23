import pandas as pd
import numpy as np
import sys
import csv
import os
from sklearn import linear_model

def main():
    # filenames = ["calibration_20230803184306"]

    # for filename in filenames:
    #     print(filename)
    #     analyze(filename + '.csv', filename + '_error_data.csv')
    

    directories = ["p01_incomplete", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10",
                   "p11", "p12", "p13", "p14_incomplete", "p15", "p16", "p17", "p18_repeat", "p19", "p20",
                   "p21_incomplete", "p22", "p23", "p24_incomplete", "p25", "p26", "p27", "p28", "p29", "p30_incomplete", 
                   "p31", "p32", "p33"]

    valid_directories = ["p02", "p03", "p04", "p06", "p07", "p08", "p09", "p10",
                   "p11", "p12", "p13", "p15", "p16", "p17", "p19", "p20",
                   "p21_incomplete", "p22", "p23", "p24_incomplete", "p25", "p26", "p27", "p28", "p29", "p30_incomplete", 
                   "p31", "p32", "p33"]

    # directories = ['sydney_pilot', 'sydney_pilot_2']

    gen_error_data(directories)
    aggregate_error_data(directories)
    average_error_data(valid_directories)

def gen_error_data(directories):
    for directory in directories:
        current_directory = os.getcwd() + "/" + directory
        final_directory = os.path.join(current_directory, r'error_data/')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        error_df = pd.DataFrame(columns=['Participant', 'Task', 'Cosine Error', 'Euclidean Error', 'Spatial Precision_Transition', 'Spatial Precision_Fixation', 'Spatial Precision_Pursuit'])

        for filename in os.scandir(directory):
            if filename.is_file():
                print(filename.path)
                date = filename.name[-18:-4]
                date = pd.to_datetime(date, format='%Y%m%d%H%M%S')
                task = filename.name[:-19]
                participant = directory
                error_data = analyze(filename.path, final_directory + filename.name.rsplit('.', 1)[0] + ".csv")
                error_df.loc[date] = [participant] + [task] + error_data

        error_path = final_directory + "error_data.csv"
        error_df.to_csv(error_path)
    
def aggregate_error_data(directories):
    final_directory = os.getcwd()
    compiled_error_data_path = final_directory + "/compiled_error_data.csv"
    compiled_error_df = pd.DataFrame()
    
    for directory in directories:
        current_directory = os.getcwd() + "/" + directory
        error_data_path = os.path.join(current_directory, r'error_data/error_data.csv')
        error_df = pd.read_csv(error_data_path)
        compiled_error_df = compiled_error_df._append(error_df, ignore_index=True)
    
    compiled_error_df.to_csv(compiled_error_data_path)
   
    aggregate_cosine_df = pd.pivot_table(compiled_error_df, index='Participant', columns='Task', values='Cosine Error', aggfunc=np.mean)
    aggregate_euclidean_df = pd.pivot_table(compiled_error_df, index='Participant', columns='Task', values='Euclidean Error', aggfunc=np.mean)
    aggregate_transition_df = pd.pivot_table(compiled_error_df.loc[compiled_error_df['Task'] == "calibration"], index='Participant', columns='Task', values='Spatial Precision_Transition', aggfunc=np.mean)
    aggregate_fixation_df = pd.pivot_table(compiled_error_df.loc[compiled_error_df['Task'] == "calibration"], index='Participant', columns='Task', values='Spatial Precision_Fixation', aggfunc=np.mean)
    aggregate_pursuit_df = pd.pivot_table(compiled_error_df.loc[compiled_error_df['Task'] == "screenStabilized_headConstrained"], index='Participant', columns='Task', values='Spatial Precision_Pursuit', aggfunc=np.mean)
    
    aggregate_cosine_df.to_csv(final_directory + "/aggregate_cosine_error_data.csv")
    aggregate_euclidean_df.to_csv(final_directory + "/aggregate_euclidean_error_data.csv")
    aggregate_transition_df.to_csv(final_directory + "/aggregate_transition_error_data.csv")
    aggregate_fixation_df.to_csv(final_directory + "/aggregate_fixation_error_data.csv")
    aggregate_pursuit_df.to_csv(final_directory + "/aggregate_pursuit_error_data.csv")

def average_error_data(participants):
    final_directory = os.getcwd()
    aggregate_cosine_error_data_path = "aggregate_cosine_error_data.csv"
    aggregate_euclidean_error_data_path = "aggregate_euclidean_error_data.csv"
    aggregate_fixation_error_data_path = "aggregate_fixation_error_data.csv"
    aggregate_transition_error_data_path = "aggregate_transition_error_data.csv"
    aggregate_pursuit_error_data_path = "aggregate_pursuit_error_data.csv"

    aggregate_data_paths = [aggregate_cosine_error_data_path, aggregate_euclidean_error_data_path, aggregate_fixation_error_data_path, aggregate_transition_error_data_path, aggregate_pursuit_error_data_path]

    for data_path in aggregate_data_paths:
        print(data_path)
        df = pd.read_csv(final_directory + "/" + data_path)
        df = df.loc[df['Participant'].isin(participants)]
        avg_error = (df.loc[:, df.columns != 'Participant']).mean(axis=0)
        print(avg_error)
        print()

    

def analyze(input_csv, output_csv):
    gaze_data = pd.read_csv(input_csv)
    # gaze_data = preprocess(gaze_data)
    gaze_data = calc_gaze_point(gaze_data)

    if("worldStabilized_sphere_VR_" in input_csv):
        gaze_data, cosine_err = calc_cosine_error_VR(gaze_data)
    else:
        gaze_data, cosine_err = calc_cosine_error(gaze_data)

    gaze_data, euclidean_err = calc_euclidean_error(gaze_data)

    if("calibration_" in input_csv):
        spatial_precision_transition = calc_spatial_precision_transition(gaze_data)
        spatial_precision_fixation = calc_spatial_precision_fixation(gaze_data)
    else:
        spatial_precision_transition = None
        spatial_precision_fixation = None
    
    if("screenStabilized_headConstrained_" in input_csv):
        spatial_precision_pursuit = calc_spatial_precision_pursuit(gaze_data)
    else:
        spatial_precision_pursuit = None

    gaze_data.to_csv(output_csv)

    error_data = [cosine_err, euclidean_err, spatial_precision_transition, spatial_precision_fixation, spatial_precision_pursuit]
    return error_data

def preprocess(df):
    result = df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]
    # result = df.loc[(df['Movement'] != "start")]

    result.reset_index(inplace=True, drop=True)
    return result

def calc_gaze_point(df):
 
    for i in range(len(df)):
        left_pos = np.array([df.loc[i, 'Gaze_Left Eye Position_x'], df.loc[i, 'Gaze_Left Eye Position_y'], df.loc[i, 'Gaze_Left Eye Position_z']])
        right_pos = np.array([df.loc[i, 'Gaze_Right Eye Position_x'], df.loc[i, 'Gaze_Right Eye Position_y'], df.loc[i, 'Gaze_Right Eye Position_z']])
        
        left_forward = np.array([df.loc[i, 'Gaze_Left Forward Vector_x'], df.loc[i, 'Gaze_Left Forward Vector_y'], df.loc[i, 'Gaze_Left Forward Vector_z']])
        right_forward = np.array([df.loc[i, 'Gaze_Right Forward Vector_x'], df.loc[i, 'Gaze_Right Forward Vector_y'], df.loc[i, 'Gaze_Right Forward Vector_z']])
        
        ball_pos = np.array([df.loc[i, 'Ball Position_x'], df.loc[i, 'Ball Position_y'], df.loc[i, 'Ball Position_z']])
        
        gaze_point = calc_vector_intersection(left_pos, right_pos, left_forward, right_forward)

        df.loc[i, 'Gaze Point_x'] = gaze_point[0][0]
        df.loc[i, 'Gaze Point_y'] = gaze_point[0][1]
        df.loc[i, 'Gaze Point_z'] = gaze_point[0][2]
        
        df.loc[i, 'Debug_Intersection'] = gaze_point[1]
        df.loc[i, 'Debug_Divergence'] = gaze_point[2]
        
    return df

def calc_vector_intersection(p1, p2, r1, r2):
    # intersection = 1
    # divergence = 0

    # p12 = p1 - p2

    # t1 = 0.0
    # t2 = 0.0

    # r2dotr2 = np.dot(r2, r2)
    # r1dotr1 = np.dot(r1, r1)
    # r1dotr2 = np.dot(r1, r2)

    # denom = pow(r1dotr2, 2) - (r1dotr1 * r2dotr2)

    # if(r1dotr2 < sys.float_info.epsilon or abs(denom) < sys.float_info.epsilon):
    #     intersection = 0
    
    # t2 = ((np.dot(p12, r1) * r2dotr2) - (np.dot(p12, r2) * r1dotr2)) / denom
    # t1 = (np.dot(p12, r2) + t2 * r1dotr1) / r1dotr2

    # if (t1 < 0 or t2 < 0):
    #     divergence = 1

    # pa = p1 + t1 * r1
    # pb = p2 + t2 * r2

    # pm = (pa + pb) / 2
    # return pm, intersection, divergence

    intersection = 1
    divergence = 0

    p21 = p1 - p2
    r2dotr2 = np.dot(r2, r2)
    r2dotr1 = np.dot(r2, r1)
    r1dotr1 = np.dot(r1, r1)

    denom = pow(r2dotr1, 2) - (r1dotr1 * r2dotr2)

    if(abs(r2dotr1) < sys.float_info.epsilon or abs(denom) < sys.float_info.epsilon):
        intersection = 0
        return [float('NaN'), float('NaN'), float('NaN')], intersection, divergence

    t2 = ((np.dot(p21, r1) * r2dotr2) - (np.dot(p21, r2) * r2dotr1)) / denom
    t1 = (np.dot(p21, r1) + (t2 * r1dotr1)) / r2dotr1
    
    if (t1 < 0 or t2 < 0):
        divergence = 1

    pa = p1 + t1 * r1
    pb = p2 + t2 * r2

    pm = (pa + pb) / 2
    return pm, intersection, divergence


def calc_cosine_error(df):
    # df['Expected_Left Gaze_x'] = df['Gaze_Left Eye Position_x'] - df['Ball Position_x']
    # df['Expected_Left Gaze_y'] = df['Gaze_Left Eye Position_y'] - df['Ball Position_y']
    # df['Expected_Left Gaze_z'] = df['Gaze_Left Eye Position_z'] - df['Ball Position_z']

    # df['Expected_Right Gaze_x'] = df['Gaze_Right Eye Position_x'] - df['Ball Position_x']
    # df['Expected_Right Gaze_y'] = df['Gaze_Right Eye Position_y'] - df['Ball Position_y']
    # df['Expected_Right Gaze_z'] = df['Gaze_Right Eye Position_z'] - df['Ball Position_z']

    # df['Left_Expected Visual Angle_x'] = np.arctan2(df['Expected_Left Gaze_x'], df['Expected_Left Gaze_z'])
    # df['Left_Expected Visual Angle_y'] = np.arctan2(df['Expected_Left Gaze_y'], df['Expected_Left Gaze_z'])

    # df['Right_Expected Visual Angle_x'] = np.arctan2(df['Expected_Right Gaze_x'], df['Expected_Right Gaze_z'])
    # df['Right_Expected Visual Angle_y'] = np.arctan2(df['Expected_Right Gaze_y'], df['Expected_Right Gaze_z'])

    # df['Left_Gaze Visual Angle_x'] = np.arctan2(df['Gaze_Left Forward Vector_x'], df['Gaze_Left Forward Vector_z'])
    # df['Left_Gaze Visual Angle_y'] = np.arctan2(df['Gaze_Left Forward Vector_y'], df['Gaze_Left Forward Vector_z'])
    
    # df['Right_Gaze Visual Angle_x'] = np.arctan2(df['Gaze_Right Forward Vector_x'], df['Gaze_Right Forward Vector_z'])
    # df['Right_Gaze Visual Angle_y'] = np.arctan2(df['Gaze_Right Forward Vector_y'], df['Gaze_Right Forward Vector_z'])

    # df['Left_Cosine Similarity'] = np.degrees(np.arccos((np.dot(np.array([df['Left_Gaze Visual Angle_x'], df['Left_Gaze Visual Angle_y']]), np.array([df['Left_Expected Visual Angle_x'], df['Left_Expected Visual Angle_y']])))/(np.linalg.norm(np.array([df['Left_Gaze Visual Angle_x'], df['Left_Gaze Visual Angle_y']])) * np.linalg.norm(np.array([df['Left_Expected Visual Angle_x'], df['Left_Expected Visual Angle_y']])))))
    # df['Right_Cosine Similarity'] = np.degrees(np.arccos((np.dot(np.array([df['Right_Gaze Visual Angle_x'], df['Right_Gaze Visual Angle_y']]), np.array([df['Right_Expected Visual Angle_x'], df['Right_Expected Visual Angle_y']])))/(np.linalg.norm(np.array([df['Right_Gaze Visual Angle_x'], df['Right_Gaze Visual Angle_y']])) * np.linalg.norm(np.array([df['Right_Expected Visual Angle_x'], df['Right_Expected Visual Angle_y']])))))
    
    # df['Avg_Cosine Similarity'] = df[['Left_Cosine Similarity', 'Right_Cosine Similarity']].mean(axis=1) 
    
    # construct expected gaze vector: center eye position from camera, and ball
    df['Expected Gaze_x'] = df['Camera_Center Eye Position_x'] - df['Ball Position_x']
    df['Expected Gaze_y'] = df['Camera_Center Eye Position_y'] - df['Ball Position_y']
    df['Expected Gaze_z'] = df['Camera_Center Eye Position_z'] - df['Ball Position_z']

    # construct actual gaze vector: center eye position by averaging eye positions, and gaze point
    df['Actual Gaze_x'] = df[['Gaze_Left Eye Position_x', 'Gaze_Right Eye Position_x']].mean(axis=1) - df['Gaze Point_x']
    df['Actual Gaze_y'] = df[['Gaze_Left Eye Position_y', 'Gaze_Right Eye Position_y']].mean(axis=1) - df['Gaze Point_y']
    df['Actual Gaze_z'] = df[['Gaze_Left Eye Position_z', 'Gaze_Right Eye Position_z']].mean(axis=1) - df['Gaze Point_z']

    # # calc visual angles
    # df['Expected Visual Angle_x'] = df.apply(lambda row: np.arctan2(row['Expected Gaze_x'], row['Expected Gaze_z']), axis=1)
    # df['Expected Visual Angle_y'] = df.apply(lambda row: np.arctan2(row['Expected Gaze_y'], row['Expected Gaze_z']), axis=1)

    # df['Actual Visual Angle_x'] = df.apply(lambda row: np.arctan2(row['Actual Gaze_x'], row['Actual Gaze_z']), axis=1)
    # df['Actual Visual Angle_y'] = df.apply(lambda row: np.arctan2(row['Actual Gaze_y'], row['Actual Gaze_z']), axis=1)

    # calc cosine similarity
    for i in range(len(df)):
        # actual_visual_angle = np.array([df.loc[i, 'Actual Visual Angle_x'], df.loc[i, 'Actual Visual Angle_y']])
        # expected_visual_angle = np.array([df.loc[i, 'Expected Visual Angle_x'], df.loc[i, 'Expected Visual Angle_y']])
        # actual_dot_expected = np.dot(actual_visual_angle, expected_visual_angle)
        # actual_visual_angle_norm = np.linalg.norm(actual_visual_angle)
        # expected_visual_angle_norm = np.linalg.norm(expected_visual_angle)
        # cosine_similarity = np.degrees(np.arccos(actual_dot_expected / (actual_visual_angle_norm * expected_visual_angle_norm)))   
        # df.loc[i, 'Cosine Similarity'] = cosine_similarity

        actual_gaze = np.array([df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z']])
        expected_gaze = np.array([df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z']])
        actual_dot_expected = np.dot(actual_gaze, expected_gaze)
        actual_gaze_norm = np.linalg.norm(actual_gaze)
        expected_gaze_norm = np.linalg.norm(expected_gaze)
        cosine_similarity = np.degrees(np.arccos(actual_dot_expected / (actual_gaze_norm * expected_gaze_norm)))   
        df.loc[i, 'Cosine Similarity'] = cosine_similarity

    
    # df['Cosine Similarity'] = df.apply(lambda row: np.degrees(np.arccos(np.dot(np.array([row['Actual Visual Angle_x'], row['Actual Visual Angle_y']]), np.array([row['Expected Visual Angle_x'], row['Expected Visual Angle_y']]))/(np.linalg.norm(np.array([row['Actual Visual Angle_x'], row['Actual Visual Angle_y']])) * np.linalg.norm(np.array([row['Expected Visual Angle_x'], df['Expected Visual Angle_y']]))))), axis=1)
    
    # avg_cosine_similarity = df['Cosine Similarity'].mean()

    avg_cosine_similarity = (df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]['Cosine Similarity']).mean()

    # print("    cosine similarity:", avg_cosine_similarity)
    return df, avg_cosine_similarity

def calc_cosine_error_VR(df):
    # df['Expected_Left Gaze_x'] = df['Gaze_Left Eye Position_x'] - df['Ball Position_x']
    # df['Expected_Left Gaze_y'] = df['Gaze_Left Eye Position_y'] - df['Ball Position_y']
    # df['Expected_Left Gaze_z'] = df['Gaze_Left Eye Position_z'] - df['Ball Position_z']

    # df['Expected_Right Gaze_x'] = df['Gaze_Right Eye Position_x'] - df['Ball Position_x']
    # df['Expected_Right Gaze_y'] = df['Gaze_Right Eye Position_y'] - df['Ball Position_y']
    # df['Expected_Right Gaze_z'] = df['Gaze_Right Eye Position_z'] - df['Ball Position_z']

    # df['Left_Expected Visual Angle_x'] = np.arctan2(df['Expected_Left Gaze_x'], df['Expected_Left Gaze_z'])
    # df['Left_Expected Visual Angle_y'] = np.arctan2(df['Expected_Left Gaze_y'], df['Expected_Left Gaze_z'])

    # df['Right_Expected Visual Angle_x'] = np.arctan2(df['Expected_Right Gaze_x'], df['Expected_Right Gaze_z'])
    # df['Right_Expected Visual Angle_y'] = np.arctan2(df['Expected_Right Gaze_y'], df['Expected_Right Gaze_z'])

    # df['Left_Gaze Visual Angle_x'] = np.arctan2(df['Gaze_Left Forward Vector_x'], df['Gaze_Left Forward Vector_z'])
    # df['Left_Gaze Visual Angle_y'] = np.arctan2(df['Gaze_Left Forward Vector_y'], df['Gaze_Left Forward Vector_z'])
    
    # df['Right_Gaze Visual Angle_x'] = np.arctan2(df['Gaze_Right Forward Vector_x'], df['Gaze_Right Forward Vector_z'])
    # df['Right_Gaze Visual Angle_y'] = np.arctan2(df['Gaze_Right Forward Vector_y'], df['Gaze_Right Forward Vector_z'])

    # df['Left_Cosine Similarity'] = np.degrees(np.arccos((np.dot(np.array([df['Left_Gaze Visual Angle_x'], df['Left_Gaze Visual Angle_y']]), np.array([df['Left_Expected Visual Angle_x'], df['Left_Expected Visual Angle_y']])))/(np.linalg.norm(np.array([df['Left_Gaze Visual Angle_x'], df['Left_Gaze Visual Angle_y']])) * np.linalg.norm(np.array([df['Left_Expected Visual Angle_x'], df['Left_Expected Visual Angle_y']])))))
    # df['Right_Cosine Similarity'] = np.degrees(np.arccos((np.dot(np.array([df['Right_Gaze Visual Angle_x'], df['Right_Gaze Visual Angle_y']]), np.array([df['Right_Expected Visual Angle_x'], df['Right_Expected Visual Angle_y']])))/(np.linalg.norm(np.array([df['Right_Gaze Visual Angle_x'], df['Right_Gaze Visual Angle_y']])) * np.linalg.norm(np.array([df['Right_Expected Visual Angle_x'], df['Right_Expected Visual Angle_y']])))))
    
    # df['Avg_Cosine Similarity'] = df[['Left_Cosine Similarity', 'Right_Cosine Similarity']].mean(axis=1) 
    
    # construct expected gaze vector: center eye position from camera, and ball
    df['Expected Gaze_x'] = df['Camera_Position_x'] - df['Ball Position_x']
    df['Expected Gaze_y'] = df['Camera_Position_y'] - df['Ball Position_y']
    df['Expected Gaze_z'] = df['Camera_Position_z'] - df['Ball Position_z']

    # construct actual gaze vector: center eye position by averaging eye positions, and gaze point
    df['Actual Gaze_x'] = df[['Gaze_Left Eye Position_x', 'Gaze_Right Eye Position_x']].mean(axis=1) - df['Gaze Point_x']
    df['Actual Gaze_y'] = df[['Gaze_Left Eye Position_y', 'Gaze_Right Eye Position_y']].mean(axis=1) - df['Gaze Point_y']
    df['Actual Gaze_z'] = df[['Gaze_Left Eye Position_z', 'Gaze_Right Eye Position_z']].mean(axis=1) - df['Gaze Point_z']

    # # calc visual angles
    # df['Expected Visual Angle_x'] = df.apply(lambda row: np.arctan2(row['Expected Gaze_x'], row['Expected Gaze_z']), axis=1)
    # df['Expected Visual Angle_y'] = df.apply(lambda row: np.arctan2(row['Expected Gaze_y'], row['Expected Gaze_z']), axis=1)

    # df['Actual Visual Angle_x'] = df.apply(lambda row: np.arctan2(row['Actual Gaze_x'], row['Actual Gaze_z']), axis=1)
    # df['Actual Visual Angle_y'] = df.apply(lambda row: np.arctan2(row['Actual Gaze_y'], row['Actual Gaze_z']), axis=1)

    # calc cosine similarity
    for i in range(len(df)):
        # actual_visual_angle = np.array([df.loc[i, 'Actual Visual Angle_x'], df.loc[i, 'Actual Visual Angle_y']])
        # expected_visual_angle = np.array([df.loc[i, 'Expected Visual Angle_x'], df.loc[i, 'Expected Visual Angle_y']])
        # actual_dot_expected = np.dot(actual_visual_angle, expected_visual_angle)
        # actual_visual_angle_norm = np.linalg.norm(actual_visual_angle)
        # expected_visual_angle_norm = np.linalg.norm(expected_visual_angle)
        # cosine_similarity = np.degrees(np.arccos(actual_dot_expected / (actual_visual_angle_norm * expected_visual_angle_norm)))   
        # df.loc[i, 'Cosine Similarity'] = cosine_similarity

        actual_gaze = np.array([df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z']])
        expected_gaze = np.array([df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z']])
        actual_dot_expected = np.dot(actual_gaze, expected_gaze)
        actual_gaze_norm = np.linalg.norm(actual_gaze)
        expected_gaze_norm = np.linalg.norm(expected_gaze)
        cosine_similarity = np.degrees(np.arccos(actual_dot_expected / (actual_gaze_norm * expected_gaze_norm)))   
        df.loc[i, 'Cosine Similarity'] = cosine_similarity

    
    # df['Cosine Similarity'] = df.apply(lambda row: np.degrees(np.arccos(np.dot(np.array([row['Actual Visual Angle_x'], row['Actual Visual Angle_y']]), np.array([row['Expected Visual Angle_x'], row['Expected Visual Angle_y']]))/(np.linalg.norm(np.array([row['Actual Visual Angle_x'], row['Actual Visual Angle_y']])) * np.linalg.norm(np.array([row['Expected Visual Angle_x'], df['Expected Visual Angle_y']]))))), axis=1)
    # avg_cosine_similarity = df['Cosine Similarity'].mean()

    avg_cosine_similarity = (df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]['Cosine Similarity']).mean()

    # print("    cosine similarity:", avg_cosine_similarity)
    return df, avg_cosine_similarity

def calc_euclidean_error(df):
    for i in range(len(df)):
        df.loc[i, 'Actual Gaze Visual Angle_x'] = np.arctan2(df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_z'])
        df.loc[i, 'Actual Gaze Visual Angle_y'] = np.arctan2(df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z'])
        
        df.loc[i, 'Expected Gaze Visual Angle_x'] = np.arctan2(df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_z'])
        df.loc[i, 'Expected Gaze Visual Angle_y'] = np.arctan2(df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z'])
        
        x_dist = df.loc[i, 'Actual Gaze Visual Angle_x'] - df.loc[i, 'Expected Gaze Visual Angle_x']
        y_dist = df.loc[i, 'Actual Gaze Visual Angle_y'] - df.loc[i, 'Expected Gaze Visual Angle_y']
        
        x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
        y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

        euclidean_error = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
        df.loc[i, 'Euclidean Error'] = euclidean_error

        # if(i == 29 or i == 859):
        #     print("frame " , i)
        #     print("actual gaze: ", df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z'])
        #     print("expected gaze: ", df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z'])

        #     print("actual x: ", np.arctan2(df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_z']))
        #     print("actual y: ", np.arctan2(df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z']))
        #     print("expected x: ", np.arctan2(df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_z']))
        #     print("expected y: ", np.arctan2(df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z']))

        #     print("x dist: ", x_dist)
        #     print("y dist: ", y_dist)

        #     print("square dist:", np.square(x_dist) + np.square(y_dist))
        #     print("sqrt: ", np.sqrt(np.square(x_dist) + np.square(y_dist)))
        #     print("in degrees: ", np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist))))
        #     print("--------------------")


    # df['Euclidean Error'] = df.apply(lambda row: np.sqrt(np.square(row['Gaze Point_x'] - row['Ball Position_x']) + np.square(row['Gaze Point_y'] - row['Ball Position_y']) + np.square(row['Gaze Point_z'] - row['Ball Position_z'])))

    # avg_euclidean_error = df['Euclidean Error'].mean()
    
    avg_euclidean_error = (df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]['Euclidean Error']).mean()    

    # print("    euclidean error:", avg_euclidean_error)
    return df, avg_euclidean_error

def calc_spatial_precision_transition(df):
    transitions = []
    rmss = np.empty(0)
    transition = False

    for i in range(len(df)):
        if pd.isnull(df.loc[i, 'Actual Gaze Visual Angle_x']):
            continue

        if df.loc[i, 'Movement'] == "transition":
            transition = True
            actual_gaze = np.array([df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']])  
            transitions.append(actual_gaze)

        if (i == len(df) - 1) or df.loc[i, 'Movement'] != "transition":
            if transition == True:
                transition = False
                euclidean_distances = np.empty(0)
                # print("transitions:", transitions)
                for i in range(1, len(transitions)):
                    x_dist = transitions[i][0] - transitions[i-1][0]
                    y_dist = transitions[i][1] - transitions[i-1][1]
            
                    # print("dist:", x_dist, y_dist)

                    x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
                    y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

                    # print("adjusted dist:", x_dist, y_dist)

                    euclidean_dist = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
                    # print(euclidean_dist)
                    euclidean_distances = np.append(euclidean_distances, euclidean_dist)

                squared_euclidean_distances = np.square(euclidean_distances)
                mean_euclidean_distances = np.mean(squared_euclidean_distances)
                rms = np.sqrt(mean_euclidean_distances)
                rmss = np.append(rmss, rms)
                transitions = []

    avg_rms = np.mean(rmss)
    return avg_rms
    # print("    transition spatial precision:", avg_rms)

def calc_spatial_precision_fixation(df):
    fixations = []
    rmss = np.empty(0)
    fixation = False

    for i in range(len(df)):
        if pd.isnull(df.loc[i, 'Actual Gaze Visual Angle_x']):
            continue

        if df.loc[i, 'Movement'] == "static":
            fixation = True
            actual_gaze = np.array([df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']])  
            fixations.append(actual_gaze)

        if (i == len(df) - 1) or df.loc[i, 'Movement'] != "static":
            if fixation == True:
                fixation = False
                euclidean_distances = np.empty(0)

                for i in range(1, len(fixations)):
                    x_dist = fixations[i][0] - fixations[i-1][0]
                    y_dist = fixations[i][1] - fixations[i-1][1]
            
                    x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
                    y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

                    euclidean_dist = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
                    euclidean_distances = np.append(euclidean_distances, euclidean_dist)

                squared_euclidean_distances = np.square(euclidean_distances)
                mean_euclidean_distances = np.mean(squared_euclidean_distances)
                rms = np.sqrt(mean_euclidean_distances)
                rmss = np.append(rmss, rms)
                fixations = []


    avg_rms = np.mean(rmss)
    # print(rmss)
    return avg_rms
    # print("    fixation spatial precision:", avg_rms)

def calc_spatial_precision_pursuit(df):
    movements = []
    rmss = np.empty(0)
    moving = False

    for i in range(len(df)):
        if pd.isnull(df.loc[i, 'Actual Gaze Visual Angle_x']):
            continue

        if df.loc[i, 'Movement'] == "moving":
            moving = True
            actual_gaze = np.array([df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']])  
            movements.append(actual_gaze)

        if (i == len(df) - 1) or df.loc[i, 'Movement'] != "moving":
            if moving == True:
                moving = False
                euclidean_distances = np.empty(0)
                # print("transitions:", transitions)
                for i in range(1, len(movements)):
                    x_dist = movements[i][0] - movements[i-1][0]
                    y_dist = movements[i][1] - movements[i-1][1]
            
                    # print("dist:", x_dist, y_dist)

                    x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
                    y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

                    # print("adjusted dist:", x_dist, y_dist)

                    euclidean_dist = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
                    # print(euclidean_dist)
                    euclidean_distances = np.append(euclidean_distances, euclidean_dist)

                squared_euclidean_distances = np.square(euclidean_distances)
                mean_euclidean_distances = np.mean(squared_euclidean_distances)
                rms = np.sqrt(mean_euclidean_distances)
                rmss = np.append(rmss, rms)
                movements = []

    avg_rms = np.mean(rmss)

    return avg_rms
    # print("    pursuit spatial precision:", avg_rms)

def calc_regression_coefficients(df):
    x1 = df[['Actual Gaze Visual Angle_x', 'Actual Gaze Visual Angle_y']]
    y1 = df['Expected Gaze Visual Angle_x']
    x2 = df[['Actual Gaze Visual Angle_x', 'Actual Gaze Visual Angle_y']]
    y2 = df['Expected Gaze Visual Angle_y']

    x_regr = linear_model.LinearRegression()
    x_regr.fit(x1.to_numpy(), y1.to_numpy())
    y_regr = linear_model.LinearRegression()
    y_regr.fit(x2.to_numpy(), y2.to_numpy())

    return [x_regr, y_regr]

def recalibrate(coefficients, df):
    


if __name__ == "__main__":
    main()