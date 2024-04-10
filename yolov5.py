import torch
import cv2
from pathlib import Path

class YOLOv5:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.classes = self.model.names

    def load_model(self, model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        return model

    def execute(self, frame):
        frame = cv2.resize(frame, (640,640))
        frame = frame.astype('float32')
        frame /= 255
        results = self.model(frame)
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        print()
        return labels, coordinates

    def plot_boxes(self,results, frame):

        """
        --> This function takes results, frame and classes
        --> results: contains labels and coordinates predicted by model on the given frame
        --> classes: contains the strting labels

        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        print(f"[INFO] Total {n} detections. . . ")
        print(f"[INFO] Looping through all detections. . . ")


        ### looping through the detections
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5: ### threshold value for detection. We are discarding everything below this value
                print(f"[INFO] Extracting BBox coordinates. . . ")
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
                text_d = self.classes[int(labels[i])]
                # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

                # if text_d == 'mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
                cv2.putText(frame, f"{text_d}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,169), 2)

                # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




        return frame


# Example usage
if __name__ == "__main__":
    model_path = "path/to/your/yolov5.pt"
    label_path = "path/to/your/labels.txt"

    yolov5 = YOLOv5(model_path, label_path)

    cap = cv2.VideoCapture(0)  # Use 0 for default camera, change to 1, 2, ... for other cameras

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        labels, coordinates = yolov5.execute(frame)
        frame = yolov5.draw_bbox(frame, labels, coordinates)

        cv2.imshow('YOLOv5 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
