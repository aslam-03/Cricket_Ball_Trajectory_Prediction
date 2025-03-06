import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model = YOLO("C:\\Users\\ASLAM\\Desktop\\Bootcamp\\drs1\\runs\\detect\\train2\\weights\\best.pt")  


video_path = "C:\\Users\\ASLAM\\Desktop\\Bootcamp\\drs1\\video\\test.mp4"  
cap = cv2.VideoCapture(video_path)

ball_positions = []


stump_x_range = (250, 370)  
pitch_y_limit = 500  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

   
    results = model(frame)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  
            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            ball_positions.append(ball_center)

         
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)


    if len(ball_positions) > 5:
        X = np.array([pos[1] for pos in ball_positions]).reshape(-1, 1)  
        Y = np.array([pos[0] for pos in ball_positions]).reshape(-1, 1)  
        model_lr = LinearRegression().fit(X, Y)
        predicted_x = model_lr.predict(np.array([[pitch_y_limit]]))[0][0]

        
        if stump_x_range[0] <= predicted_x <= stump_x_range[1]:
            decision = "OUT"
            color = (0, 0, 255)  
        else:
            decision = "NOT OUT"
            color = (0, 255, 0) 

        
        cv2.putText(frame, decision, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

   
    cv2.imshow("DRS System", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
