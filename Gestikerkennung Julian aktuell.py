# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:38:52 2022

@author: Julian
"""

import cv2
import mediapipe as mp
import numpy as np
import math
#import pandas as pd


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
# midpoint of two image points 
def midpoint(point1 ,point2):
    return (point1[0] + point2[0])/2,(point1[1] + point2[1])/2

def abs_distance (point1,point2,point3):
    mid = midpoint(point1,point2)
    res = abs(mid[1]-point3[1])
    return res

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Nod counter variables
counter_up_down = counter_right_left = 0
counter = counter_s = 0 
sequence_angle= sequence_dis_r = sequence_dis_l = sequence_dis_nose_shoulder = []

rel_dis=[]

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
             # Get coordinates
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            ear_r = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            ear_l = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            
            
            # Calculate angle
            angle = calculate_angle(shoulder_l, nose, shoulder_r)
            
            # Visualize angle
            # cv2.putText(image, str(angle), 
            #                tuple(np.multiply(nose, [640, 480]).astype(int)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #                     )
            
            dis_r = euclidean_distance(nose , ear_r)
            dis_l = euclidean_distance(nose , ear_l)
            dis_nose_shoulder = abs_distance(shoulder_l, shoulder_r, nose)
        
            sequence_angle.append(angle)
            sequence_angle = sequence_angle[-35:]
            
            sequence_dis_r.append(dis_r)
            sequence_dis_r = sequence_dis_r[-35:]
            
            sequence_dis_l.append(dis_l)
            sequence_dis_l = sequence_dis_l[-35:]
            
            sequence_dis_nose_shoulder.append(dis_nose_shoulder)
            sequence_dis_nose_shoulder = sequence_dis_nose_shoulder[-35:]
            
            if len(sequence_angle) == 35: 
                for i in range(len(sequence_angle)-1):
                                        
       
                    #print(abs((sequence_angle[0] - sequence_angle[i+1]) / sequence_angle[i]))
                    if abs((sequence_angle[i] - sequence_angle[i+1]) / sequence_angle[i])>0.08:
                        counter_up_down +=1
                        
                    #print(abs((sequence_dis_l[i] - sequence_dis_l[i+1]) / sequence_dis_l[i]))    
                    
                    if abs((sequence_dis_l[i] - sequence_dis_l[i+1]) / sequence_dis_l[i])>0.20 and abs((sequence_dis_nose_shoulder[i] - sequence_dis_nose_shoulder[i+1]) / sequence_dis_nose_shoulder[i])<0.05:
                        counter_right_left+=1
                        
                    if abs((sequence_dis_r[i] - sequence_dis_r[i+1]) / sequence_dis_r[i])>0.20 and abs((sequence_dis_nose_shoulder[i] - sequence_dis_nose_shoulder[i+1]) / sequence_dis_nose_shoulder[i])<0.05:
                        counter_right_left+=1
                        
                if counter_up_down > 1: #and counter_right_left <= 0:
                    counter += 1
                    counter_up_down = 0
                    del sequence_angle[:]
                    
                if counter_right_left > 1: # and  counter_up_down <= 1 #and :
                    counter_s += 1
                    counter_right_left = 0
                    del sequence_dis_r[:]
                    del sequence_dis_l[:]

                
        except:
            pass
        
        # # Render curl counter
        # # Setup status box
        # cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        #Rep data
        cv2.putText(image, 'NICKEN', (15,25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'SCHUETTELN', (250,25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_s), 
                    (250,75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
       
        
        # # Stage data
        # cv2.putText(image, 'STAGE', (65,12), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, stage, 
        #             (60,60), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        #Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                  )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()