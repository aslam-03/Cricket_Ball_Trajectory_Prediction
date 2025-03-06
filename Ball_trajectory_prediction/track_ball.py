import cv2
from ultralytics import YOLO


model = YOLO("C:\\Users\\ASLAM\\Desktop\\Bootcamp\\drs1\\runs\\detect\\train2\\weights\\best.pt")

video_path = "C:\\Users\\ASLAM\\Desktop\\Bootcamp\\drs1\\video\\test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))


    results = model.predict(frame)  
    for result in results[0].boxes.data:  
        x1, y1, x2, y2, conf, cls = result.tolist()
        if int(cls) == 0:  
           
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

 
    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
