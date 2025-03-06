from ultralytics import YOLO


model = YOLO("yolov8n.pt")  


model.train(
    data="C:/Users/ASLAM/Desktop/Bootcamp/drs1/data.yaml",
    epochs=30,
    imgsz=416,
    workers=2,
    device="cuda:0",  
    half=True
)
