import pandas as pd
import numpy as np
import sys
import csv

def main():
    gaze_data = pd.read_csv('../Assets/calibration_20230614180750.csv')
    gaze_data['Gaze Point_x'] = ''
    gaze_data['Gaze Point_y'] = ''
    gaze_data['Gaze Point_z'] = ''
    gaze_data['Debug_Intersection'] = ''
    gaze_data['Debug_Divergence'] = ''
    
    for i in range(len(gaze_data)):
        left_pos = np.array([gaze_data.loc[i, 'Gaze_Left Eye Position_x'], gaze_data.loc[i, 'Gaze_Left Eye Position_y'], gaze_data.loc[i, 'Gaze_Left Eye Position_z']])
        right_pos = np.array([gaze_data.loc[i, 'Gaze_Right Eye Position_x'], gaze_data.loc[i, 'Gaze_Right Eye Position_y'], gaze_data.loc[i, 'Gaze_Right Eye Position_z']])
        
        left_forward = np.array([gaze_data.loc[i, 'Gaze_Left Forward Vector_x'], gaze_data.loc[i, 'Gaze_Left Forward Vector_y'], gaze_data.loc[i, 'Gaze_Left Forward Vector_z']])
        right_forward = np.array([gaze_data.loc[i, 'Gaze_Right Forward Vector_x'], gaze_data.loc[i, 'Gaze_Right Forward Vector_y'], gaze_data.loc[i, 'Gaze_Right Forward Vector_z']])
        
        ball_pos = np.array([gaze_data.loc[i, 'Ball Position_x'], gaze_data.loc[i, 'Ball Position_y'], gaze_data.loc[i, 'Ball Position_z']])
        
        gaze_point = calc_vector_intersection(left_pos, right_pos, left_forward, right_forward)

        gaze_data.loc[i, 'Gaze Point_x'] = gaze_point[0][0]
        gaze_data.loc[i, 'Gaze Point_y'] = gaze_point[0][1]
        gaze_data.loc[i, 'Gaze Point_z'] = gaze_point[0][2]
        
        gaze_data.loc[i, 'Debug_Intersection'] = gaze_point[1]
        gaze_data.loc[i, 'Debug_Divergence'] = gaze_point[2]
        
    error_data = gaze_data[['Frame', 'Movement', 'Gaze Point_x', 'Gaze Point_y', 'Gaze Point_z', 'Debug_Intersection', 'Debug_Divergence', 'Ball Position_x', 'Ball Position_y', 'Ball Position_z']]
    print(error_data)
    error_data.to_csv('error_data.csv')

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

def calc_gaze_error():
    

if __name__ == "__main__":
    main()