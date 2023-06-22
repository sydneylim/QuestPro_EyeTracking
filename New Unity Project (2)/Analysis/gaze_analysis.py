import pandas as pd
import numpy as np
import sys
import csv

def main():
    gaze_data = pd.read_csv('../Assets/calibration_20230614180750.csv')
    gaze_data = calc_gaze_point(gaze_data)
    gaze_data = calc_cosine_error(gaze_data)
    gaze_data = calc_euclidean_error(gaze_data)

    error_data = gaze_data[['Frame', 'Movement', 'Gaze Point_x', 'Gaze Point_y', 'Gaze Point_z', 'Debug_Intersection', 'Debug_Divergence', 'Ball Position_x', 'Ball Position_y', 'Ball Position_z', 'Left_Cosine Similarity', 'Right_Cosine Similarity', 'Avg_Cosine Similarity', 'Euclidean Error']]

    error_data.to_csv('error_data.csv')

def calc_gaze_point(df):
    df['Gaze Point_x'] = ''
    df['Gaze Point_y'] = ''
    df['Gaze Point_z'] = ''
    df['Debug_Intersection'] = ''
    df['Debug_Divergence'] = ''
    
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
    err_intersection = 1
    err_divergence = 0

    p12 = p1 - p2

    t1 = 0.0
    t2 = 0.0

    r2dotr2 = np.dot(r2, r2)
    r1dotr1 = np.dot(r1, r1)
    r1dotr2 = np.dot(r1, r2)

    denom = pow(r1dotr2, 1) - (r1dotr1 * r2dotr2);

    if(r1dotr2 < sys.float_info.epsilon or abs(denom) < sys.float_info.epsilon):
        err_intersection = 0
    
    t2 = ((np.dot(p12, r1) * r2dotr2) - (np.dot(p12, r2) * r1dotr2)) / denom;
    t1 = (np.dot(p12, r2) + t2 * r1dotr1) / r1dotr2;

    if (t1 < 0 or t2 < 0):
        err_divergence = 1

    pa = p1 + t1 * r1
    pb = p2 + t2 * r2

    pm = (pa + pb) / 2
    return pm, err_intersection, err_divergence

def calc_cosine_error(df):
    df['Expected_Left Gaze_x'] = df['Gaze_Left Eye Position_x'] - df['Ball Position_x']
    df['Expected_Left Gaze_y'] = df['Gaze_Left Eye Position_y'] - df['Ball Position_y']
    df['Expected_Left Gaze_z'] = df['Gaze_Left Eye Position_z'] - df['Ball Position_z']

    df['Expected_Right Gaze_x'] = df['Gaze_Right Eye Position_x'] - df['Ball Position_x']
    df['Expected_Right Gaze_y'] = df['Gaze_Right Eye Position_y'] - df['Ball Position_y']
    df['Expected_Right Gaze_z'] = df['Gaze_Right Eye Position_z'] - df['Ball Position_z']

    df['Left_Expected Visual Angle_x'] = np.arctan2(df['Expected_Left Gaze_x'], df['Expected_Left Gaze_z'])
    df['Left_Expected Visual Angle_y'] = np.arctan2(df['Expected_Left Gaze_y'], df['Expected_Left Gaze_z'])

    df['Right_Expected Visual Angle_x'] = np.arctan2(df['Expected_Right Gaze_x'], df['Expected_Right Gaze_z'])
    df['Right_Expected Visual Angle_y'] = np.arctan2(df['Expected_Right Gaze_y'], df['Expected_Right Gaze_z'])

    df['Left_Gaze Visual Angle_x'] = np.arctan2(df['Gaze_Left Forward Vector_x'], df['Gaze_Left Forward Vector_z'])
    df['Left_Gaze Visual Angle_y'] = np.arctan2(df['Gaze_Left Forward Vector_y'], df['Gaze_Left Forward Vector_z'])
    
    df['Right_Gaze Visual Angle_x'] = np.arctan2(df['Gaze_Right Forward Vector_x'], df['Gaze_Right Forward Vector_z'])
    df['Right_Gaze Visual Angle_y'] = np.arctan2(df['Gaze_Right Forward Vector_y'], df['Gaze_Right Forward Vector_z'])

    df['Left_Cosine Similarity'] = np.degrees(np.arccos((np.dot(np.array([df['Left_Gaze Visual Angle_x'], df['Left_Gaze Visual Angle_y']]), np.array([df['Left_Expected Visual Angle_x'], df['Left_Expected Visual Angle_y']])))/(np.linalg.norm(np.array([df['Left_Gaze Visual Angle_x'], df['Left_Gaze Visual Angle_y']])) * np.linalg.norm(np.array([df['Left_Expected Visual Angle_x'], df['Left_Expected Visual Angle_y']])))))
    df['Right_Cosine Similarity'] = np.degrees(np.arccos((np.dot(np.array([df['Right_Gaze Visual Angle_x'], df['Right_Gaze Visual Angle_y']]), np.array([df['Right_Expected Visual Angle_x'], df['Right_Expected Visual Angle_y']])))/(np.linalg.norm(np.array([df['Right_Gaze Visual Angle_x'], df['Right_Gaze Visual Angle_y']])) * np.linalg.norm(np.array([df['Right_Expected Visual Angle_x'], df['Right_Expected Visual Angle_y']])))))
    
    df['Avg_Cosine Similarity'] = df[['Left_Cosine Similarity', 'Right_Cosine Similarity']].mean(axis=1) 
    
    cosine_similarity = df['Avg_Cosine Similarity'].mean()
    print(cosine_similarity)
    return df

def calc_euclidean_error(df):
    df['Euclidean Error'] = np.sqrt(np.square(df['Gaze Point_x'] - df['Ball Position_x']) + np.square(df['Gaze Point_y'] - df['Ball Position_y']) + np.square(df['Gaze Point_z'] - df['Ball Position_z']))
    
    euclidean_error = df['Euclidean Error'].mean()
    print(euclidean_error)
    return df

if __name__ == "__main__":
    main()