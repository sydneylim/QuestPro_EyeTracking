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
    

    directories = ["p03", "p04", "p05", "p06_incomplete", "p07", "p08", "p09", "p10",
                   "p11", "p12", "p13", "p14_incomplete", "p15", "p16", "p17", "p18_repeat", "p19", "p20",
                   "p21_incomplete", "p22", "p23", "p24_incomplete", "p25", "p26", "p27", "p28", "p29", "p30_incomplete", 
                   "p31", "p32", "p33"]

    valid_directories = ["p03", "p04", "p05", "p06_incomplete", "p07", "p08", "p09", "p10",
                   "p11", "p12", "p13", "p15", "p16", "p17", "p19", "p20",
                   "p21_incomplete", "p22", "p23", "p24_incomplete", "p25", "p26", "p27", "p28", "p29", "p30_incomplete", 
                   "p31", "p32", "p33"]


    # directories = ['p03']
    # valid_directories = ['p03']
    
    #gen_error_data(directories)
    # aggregate_error_data(directories, valid_directories)
    average_error_data(valid_directories)

# gen error data per participant
def gen_error_data(directories):
    for directory in directories:
        current_directory = os.getcwd() + "/" + directory
        final_directory = os.path.join(current_directory, r'error_data/')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        error_df = pd.DataFrame(columns=['Participant', 'Task', 'Trial', 'Cosine Error', 'Static Cosine Error', 'Moving Cosine Error', 'Euclidean Error', 'Static Euclidean Error', 'Moving Euclidean Error', 'Spatial Precision_Transition', 'Spatial Precision_Fixation', 'Spatial Precision_Pursuit',
                                         'Recalibrated_Euclidean Error', 'Recalibrated_Static Euclidean Error', 'Recalibrated_Moving Euclidean Error', 'Recalibrated_Spatial Precision_Transition', 'Recalibrated_Spatial Precision_Fixation', 'Recalibrated_Spatial Precision_Pursuit'])

        # calibration first to get recalibration coefficients
        compiled_calibration_df = pd.DataFrame()

        for filename in os.scandir(directory):
            if filename.name[:-21] == "calibration":
                calibration_df = pd.read_csv(filename.path)
                compiled_calibration_df = compiled_calibration_df._append(calibration_df, ignore_index=True)
        
        compiled_calibration_df = calc_gaze_point(compiled_calibration_df)
        compiled_calibration_df = calc_gaze_vectors(compiled_calibration_df, False)
        compiled_calibration_df = calc_gaze_visual_angles(compiled_calibration_df)
        compiled_calibration_df = compiled_calibration_df.dropna(ignore_index=True)
        [x_coeff, y_coeff] = calc_regression_coefficients(compiled_calibration_df)
        
        compiled_calibration_df.to_csv(final_directory + "compiled_calibration_data.csv")

        for filename in os.scandir(directory):
            if filename.is_file():
                print(filename.path)
                date = pd.to_datetime(filename.name[-20:-6], format='%Y%m%d%H%M%S')
                task = filename.name[:-21]
                trial = filename.name[-5:-4]
                participant = directory
                print(date, task, trial)
                error_data = analyze(x_coeff, y_coeff, filename.path, final_directory + filename.name.rsplit('.', 1)[0] + ".csv")
                error_df.loc[date] = [participant] + [task] + [trial] + error_data

        error_path = final_directory + "error_data.csv"
        error_df.to_csv(error_path)

# compile all error data into 1 file, separate types of error data into separate files
def aggregate_error_data(directories, participants):
    final_directory = os.getcwd()
    compiled_error_data_path = final_directory + "/compiled_error_data.csv"
    compiled_error_df = pd.DataFrame()
    
    for directory in directories:
        current_directory = os.getcwd() + "/" + directory
        error_data_path = os.path.join(current_directory, r'error_data/error_data.csv')
        error_df = pd.read_csv(error_data_path)
        compiled_error_df = compiled_error_df._append(error_df, ignore_index=True)
    
    compiled_error_df.to_csv(compiled_error_data_path)
    
    aggregate_error_df = compiled_error_df.drop(compiled_error_df.columns[0], axis=1)
    aggregate_error_df = aggregate_error_df.loc[aggregate_error_df['Participant'].isin(participants)]
    aggregate_error_df = aggregate_error_df.groupby(['Participant', 'Task']).mean()
    aggregate_error_df.to_csv(final_directory + "/aggregate_error_data.csv")

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

    aggregate_recalibrated_euclidean_df = pd.pivot_table(compiled_error_df, index='Participant', columns='Task', values='Recalibrated_Euclidean Error', aggfunc=np.mean)
    aggregate_recalibrated_transition_df = pd.pivot_table(compiled_error_df.loc[compiled_error_df['Task'] == "calibration"], index='Participant', columns='Task', values='Recalibrated_Spatial Precision_Transition', aggfunc=np.mean)
    aggregate_recalibrated_fixation_df = pd.pivot_table(compiled_error_df.loc[compiled_error_df['Task'] == "calibration"], index='Participant', columns='Task', values='Recalibrated_Spatial Precision_Fixation', aggfunc=np.mean)
    aggregate_recalibrated_pursuit_df = pd.pivot_table(compiled_error_df.loc[compiled_error_df['Task'] == "screenStabilized_headConstrained"], index='Participant', columns='Task', values='Recalibrated_Spatial Precision_Pursuit', aggfunc=np.mean)
    
    aggregate_recalibrated_euclidean_df.to_csv(final_directory + "/aggregate_recalibrated_euclidean_error_data.csv")
    aggregate_recalibrated_transition_df.to_csv(final_directory + "/aggregate_recalibrated_transition_error_data.csv")
    aggregate_recalibrated_fixation_df.to_csv(final_directory + "/aggregate_recalibrated_fixation_error_data.csv")
    aggregate_recalibrated_pursuit_df.to_csv(final_directory + "/aggregate_recalibrated_pursuit_error_data.csv")

# print average error values per task
def average_error_data(participants):
    final_directory = os.getcwd()
    aggregate_cosine_error_data_path = "aggregate_cosine_error_data.csv"
    aggregate_euclidean_error_data_path = "aggregate_euclidean_error_data.csv"
    aggregate_fixation_error_data_path = "aggregate_fixation_error_data.csv"
    aggregate_transition_error_data_path = "aggregate_transition_error_data.csv"
    aggregate_pursuit_error_data_path = "aggregate_pursuit_error_data.csv"

    aggregate_recalibrated_euclidean_error_data_path = "aggregate_recalibrated_euclidean_error_data.csv"
    aggregate_recalibrated_fixation_error_data_path = "aggregate_recalibrated_fixation_error_data.csv"
    aggregate_recalibrated_transition_error_data_path = "aggregate_recalibrated_transition_error_data.csv"
    aggregate_recalibrated_pursuit_error_data_path = "aggregate_recalibrated_pursuit_error_data.csv"

    aggregate_data_paths = [aggregate_cosine_error_data_path, aggregate_euclidean_error_data_path, aggregate_fixation_error_data_path, aggregate_transition_error_data_path, aggregate_pursuit_error_data_path,
                            aggregate_recalibrated_euclidean_error_data_path, aggregate_recalibrated_fixation_error_data_path, aggregate_recalibrated_transition_error_data_path, aggregate_recalibrated_pursuit_error_data_path]

    print()
    for data_path in aggregate_data_paths:
        print(data_path)
        df = pd.read_csv(final_directory + "/" + data_path)
        df = df.loc[df['Participant'].isin(participants)]
        avg_error = (df.loc[:, df.columns != 'Participant']).mean(axis=0)
        print(avg_error)
        print()

def analyze(x_coeff, y_coeff, input_csv, output_csv):
    gaze_data = pd.read_csv(input_csv)
    
    # calc gaze intersection point
    gaze_data = calc_gaze_point(gaze_data)

    # calc gaze vector
    if("worldStabilized_sphere_VR_" in input_csv):
        gaze_data = calc_gaze_vectors(gaze_data, True)
    else:
        gaze_data = calc_gaze_vectors(gaze_data, False)

    # calc visual angle
    gaze_data = calc_gaze_visual_angles(gaze_data)

    # drop rows with blanks
    gaze_data = gaze_data.dropna(ignore_index=True)

    # calc recalibrated gaze data
    gaze_data = recalibrate(x_coeff, y_coeff, gaze_data)

    # calc cosine and euclidean errors
    gaze_data, cosine_err, static_cosine_err, moving_cosine_err = calc_cosine_error(gaze_data)
    gaze_data, euclidean_err, static_euclidean_err, moving_euclidean_err = calc_euclidean_error(gaze_data, False)

    # calc spatial precisions
    if("calibration_" in input_csv):
        spatial_precision_transition = calc_spatial_precision_transition(gaze_data, False)
        spatial_precision_fixation = calc_spatial_precision_fixation(gaze_data, False)
    else:
        spatial_precision_transition = None
        spatial_precision_fixation = None
    
    if("screenStabilized_headConstrained_" in input_csv):
        spatial_precision_pursuit = calc_spatial_precision_pursuit(gaze_data, False)
    else:
        spatial_precision_pursuit = None

    # calc recalibrated error data
    gaze_data, recalibrated_euclidean_err, recalibrated_static_euclidean_err, recalibrated_moving_euclidean_err = calc_euclidean_error(gaze_data, True)

    if("calibration_" in input_csv):
        recalibrated_spatial_precision_transition = calc_spatial_precision_transition(gaze_data, True)
        recalibrated_spatial_precision_fixation = calc_spatial_precision_fixation(gaze_data, True)
    else:
        recalibrated_spatial_precision_transition = None
        recalibrated_spatial_precision_fixation = None
    
    if("screenStabilized_headConstrained_" in input_csv):
        recalibrated_spatial_precision_pursuit = calc_spatial_precision_pursuit(gaze_data, True)
    else:
        recalibrated_spatial_precision_pursuit = None

    # export gaze data
    gaze_data.to_csv(output_csv)

    # return error_data
    error_data = [cosine_err, static_cosine_err, moving_cosine_err, euclidean_err, static_euclidean_err, moving_euclidean_err, spatial_precision_transition, spatial_precision_fixation, spatial_precision_pursuit,
                  recalibrated_euclidean_err, recalibrated_static_euclidean_err, recalibrated_moving_euclidean_err, recalibrated_spatial_precision_transition, recalibrated_spatial_precision_fixation, recalibrated_spatial_precision_pursuit]
    return error_data

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

def calc_gaze_vectors(df, vr):
    if vr == False:
        # construct expected gaze vector: center eye position from camera, and ball
        df['Expected Gaze_x'] = df['Camera_Center Eye Position_x'] - df['Ball Position_x']
        df['Expected Gaze_y'] = df['Camera_Center Eye Position_y'] - df['Ball Position_y']
        df['Expected Gaze_z'] = df['Camera_Center Eye Position_z'] - df['Ball Position_z']

        # construct actual gaze vector: center eye position by averaging eye positions, and gaze point
        df['Actual Gaze_x'] = df[['Gaze_Left Eye Position_x', 'Gaze_Right Eye Position_x']].mean(axis=1) - df['Gaze Point_x']
        df['Actual Gaze_y'] = df[['Gaze_Left Eye Position_y', 'Gaze_Right Eye Position_y']].mean(axis=1) - df['Gaze Point_y']
        df['Actual Gaze_z'] = df[['Gaze_Left Eye Position_z', 'Gaze_Right Eye Position_z']].mean(axis=1) - df['Gaze Point_z']

    else:
        # construct expected gaze vector: center eye position from camera, and ball
        df['Expected Gaze_x'] = df['Camera_Position_x'] - df['Ball Position_x']
        df['Expected Gaze_y'] = df['Camera_Position_y'] - df['Ball Position_y']
        df['Expected Gaze_z'] = df['Camera_Position_z'] - df['Ball Position_z']

        # construct actual gaze vector: center eye position by averaging eye positions, and gaze point
        df['Actual Gaze_x'] = df[['Gaze_Left Eye Position_x', 'Gaze_Right Eye Position_x']].mean(axis=1) - df['Gaze Point_x']
        df['Actual Gaze_y'] = df[['Gaze_Left Eye Position_y', 'Gaze_Right Eye Position_y']].mean(axis=1) - df['Gaze Point_y']
        df['Actual Gaze_z'] = df[['Gaze_Left Eye Position_z', 'Gaze_Right Eye Position_z']].mean(axis=1) - df['Gaze Point_z']

    return df

def calc_gaze_visual_angles(df):
    for i in range(len(df)):
        df.loc[i, 'Actual Gaze Visual Angle_x'] = np.arctan2(df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_z'])
        df.loc[i, 'Actual Gaze Visual Angle_y'] = np.arctan2(df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z'])
        
        df.loc[i, 'Expected Gaze Visual Angle_x'] = np.arctan2(df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_z'])
        df.loc[i, 'Expected Gaze Visual Angle_y'] = np.arctan2(df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z'])

    return df

# input calibration df
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

def recalibrate(x_coeff, y_coeff, df):
    for i in range(len(df)):
        df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_x'] = x_coeff.predict([[df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']]])[0]
        df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_y'] = y_coeff.predict([[df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']]])[0]

    return df

def calc_cosine_error(df):
    # calc cosine similarity
    for i in range(len(df)):
        actual_gaze = np.array([df.loc[i, 'Actual Gaze_x'], df.loc[i, 'Actual Gaze_y'], df.loc[i, 'Actual Gaze_z']])
        expected_gaze = np.array([df.loc[i, 'Expected Gaze_x'], df.loc[i, 'Expected Gaze_y'], df.loc[i, 'Expected Gaze_z']])
        actual_dot_expected = np.dot(actual_gaze, expected_gaze)
        actual_gaze_norm = np.linalg.norm(actual_gaze)
        expected_gaze_norm = np.linalg.norm(expected_gaze)
        cosine_similarity = np.degrees(np.arccos(actual_dot_expected / (actual_gaze_norm * expected_gaze_norm)))   
        df.loc[i, 'Cosine Similarity'] = cosine_similarity

    avg_cosine_similarity = (df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]['Cosine Similarity']).mean()
    avg_static_cosine_similarity = (df.loc[(df['Movement'] == "static")]['Cosine Similarity']).mean()
    avg_moving_cosine_similarity = (df.loc[(df['Movement'] == "moving")]['Cosine Similarity']).mean()

    # print("    cosine similarity:", avg_cosine_similarity)
    return df, avg_cosine_similarity, avg_static_cosine_similarity, avg_moving_cosine_similarity

def calc_euclidean_error(df, recalibrated):
    if recalibrated == False:
        for i in range(len(df)):        
            x_dist = df.loc[i, 'Actual Gaze Visual Angle_x'] - df.loc[i, 'Expected Gaze Visual Angle_x']
            y_dist = df.loc[i, 'Actual Gaze Visual Angle_y'] - df.loc[i, 'Expected Gaze Visual Angle_y']
            
            x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
            y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

            euclidean_error = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
            df.loc[i, 'Euclidean Error'] = euclidean_error
        
        avg_euclidean_error = (df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]['Euclidean Error']).mean()    
        avg_static_euclidean_error = (df.loc[(df['Movement'] == "static")]['Euclidean Error']).mean()
        avg_moving_euclidean_error = (df.loc[(df['Movement'] == "moving")]['Euclidean Error']).mean()    

    else:
        for i in range(len(df)):        
            x_dist = df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_x'] - df.loc[i, 'Expected Gaze Visual Angle_x']
            y_dist = df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_y'] - df.loc[i, 'Expected Gaze Visual Angle_y']
            
            x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
            y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

            euclidean_error = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
            df.loc[i, 'Recalibrated_Euclidean Error'] = euclidean_error
        
        avg_euclidean_error = (df.loc[(df['Movement'] != "start") & (df['Movement'] != "transition")]['Recalibrated_Euclidean Error']).mean()    
        avg_static_euclidean_error = (df.loc[(df['Movement'] == "static")]['Recalibrated_Euclidean Error']).mean()
        avg_moving_euclidean_error = (df.loc[(df['Movement'] == "moving")]['Recalibrated_Euclidean Error']).mean()

    # print("    euclidean error:", avg_euclidean_error)
    return df, avg_euclidean_error, avg_static_euclidean_error, avg_moving_euclidean_error

def calc_spatial_precision_transition(df, recalibrated):
    transitions = []
    rmss = np.empty(0)
    transition = False

    for i in range(len(df)):
        # if pd.isnull(df.loc[i, 'Actual Gaze Visual Angle_x']):
        #     continue

        if df.loc[i, 'Movement'] == "transition":
            transition = True

            if recalibrated == False:
                actual_gaze = np.array([df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']])  
            else: 
                actual_gaze = np.array([df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_x'], df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_y']])  

            transitions.append(actual_gaze)

        if (i == len(df) - 1) or df.loc[i, 'Movement'] != "transition":
            if transition == True:
                transition = False
                euclidean_distances = np.empty(0)
                # print("transitions:", transitions)
                for i in range(1, len(transitions)):
                    x_dist = transitions[i][0] - transitions[i-1][0]
                    y_dist = transitions[i][1] - transitions[i-1][1]
            
                    x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
                    y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

                    euclidean_dist = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
                    euclidean_distances = np.append(euclidean_distances, euclidean_dist)

                squared_euclidean_distances = np.square(euclidean_distances)
                mean_euclidean_distances = np.mean(squared_euclidean_distances)
                rms = np.sqrt(mean_euclidean_distances)
                rmss = np.append(rmss, rms)
                transitions = []

    avg_rms = np.mean(rmss)
    return avg_rms
    # print("    transition spatial precision:", avg_rms)

def calc_spatial_precision_fixation(df, recalibrated):
    fixations = []
    rmss = np.empty(0)
    fixation = False

    for i in range(len(df)):
        # if pd.isnull(df.loc[i, 'Actual Gaze Visual Angle_x']):
        #     continue

        if df.loc[i, 'Movement'] == "static":
            fixation = True

            if recalibrated == False:
                actual_gaze = np.array([df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']])  
            else: 
                actual_gaze = np.array([df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_x'], df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_y']])  
            
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

def calc_spatial_precision_pursuit(df, recalibrated):
    movements = []
    rmss = np.empty(0)
    moving = False

    for i in range(len(df)):
        # if pd.isnull(df.loc[i, 'Actual Gaze Visual Angle_x']):
        #     continue

        if df.loc[i, 'Movement'] == "moving":
            moving = True

            if recalibrated == False:
                actual_gaze = np.array([df.loc[i, 'Actual Gaze Visual Angle_x'], df.loc[i, 'Actual Gaze Visual Angle_y']])  
            else: 
                actual_gaze = np.array([df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_x'], df.loc[i, 'Recalibrated_Actual Gaze Visual Angle_y']])  

            movements.append(actual_gaze)

        if (i == len(df) - 1) or df.loc[i, 'Movement'] != "moving":
            if moving == True:
                moving = False
                euclidean_distances = np.empty(0)
                for i in range(1, len(movements)):
                    x_dist = movements[i][0] - movements[i-1][0]
                    y_dist = movements[i][1] - movements[i-1][1]
            
                    x_dist = (x_dist + np.pi) % (np.pi*2) - np.pi
                    y_dist = (y_dist + np.pi) % (np.pi*2) - np.pi

                    euclidean_dist = np.degrees(np.sqrt(np.square(x_dist) + np.square(y_dist)))
                    euclidean_distances = np.append(euclidean_distances, euclidean_dist)

                squared_euclidean_distances = np.square(euclidean_distances)
                mean_euclidean_distances = np.mean(squared_euclidean_distances)
                rms = np.sqrt(mean_euclidean_distances)
                rmss = np.append(rmss, rms)
                movements = []

    avg_rms = np.mean(rmss)

    return avg_rms
    # print("    pursuit spatial precision:", avg_rms)


if __name__ == "__main__":
    main()