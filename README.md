# Cricket_Ball_Trajectory_Prediction

This project uses YOLO (You Only Look Once) object detection and linear regression to track a cricket ball's trajectory and predict its impact point. It helps in making LBW (Leg Before Wicket) decisions based on ball movement.

#Features

Real-time ball detection using YOLO.

Trajectory tracking with a line showing ball movement.

Impact point prediction using Linear Regression.

Live "OUT" or "NOT OUT" decision display.

Smooth, continuous video playback without pauses.

#Requirements

Python 3.x

OpenCV (pip install opencv-python)

NumPy (pip install numpy)

Ultralytics YOLO (pip install ultralytics)

Scikit-learn (pip install scikit-learn)


#How to Run

Download the trained YOLO model (best.pt) and place it in the project directory.

Put a test video inside the video folder.

Run the script:

python main.py

Press 'Q' to exit the video window.

#Output

A red dot on detected ball positions.

A yellow line showing the ball's path.

A blue dot at the predicted impact point.

"LBW Out!" or "Not Out" displayed live on the video.

#Notes

Adjust the stumps_x value in the code to match the wicket position in your video.

If ball detection is inaccurate, try retraining YOLO on a better dataset.



