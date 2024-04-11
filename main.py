import cv2

from yolov5 import YOLOv5
import config
yolov5 = YOLOv5(model_path = config.MODEL_PATH)

cap = cv2.VideoCapture(0)  # Use 0 for default camera, change to 1, 2, ... for other cameras

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    outputs = yolov5.execute(frame)
    frame = yolov5.plot_boxes(outputs, frame)
    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()