import time
import numpy as np
import onnxruntime
import cv2

import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image to new_shape while maintaining aspect ratio
    h, w = im.shape[:2]  # current height, width
    target_h, target_w = new_shape

    # Calculate aspect ratio
    aspect_ratio = min(target_w / w, target_h / h)

    # Calculate new width and height
    new_w = int(w * aspect_ratio)
    new_h = int(h * aspect_ratio)

    # Resize image
    resized_im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas of the target size
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    # Calculate coordinates to paste resized image in the center
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_im

    return canvas

# Example usage:
# im = cv2.imread("image.jpg")
# letterboxed_im = letterbox(im, new_shape=(640, 640), color=(114, 114, 114))

names = {
  0: "normal",
  1: "drowsy",
  2: "drowsy#2",
  3: "yawning",
 }
def postprocess_yolov5(output_data, confidence_threshold=0.25, nms_threshold=0.45):
    detections = []
    grid_size = output_data.shape[1]

    x_min = output_data[0, :, 0]
    y_min = output_data[0, :, 1]
    x_max = output_data[0, :, 2]
    y_max = output_data[0, :, 3]
    confidence = output_data[0, :, 4]
    class_probs = output_data[0, :, 5:]

    class_id = np.argmax(class_probs, axis=1)
    class_prob = np.max(class_probs * confidence[:, np.newaxis], axis=1)

    width = x_max
    height = y_max

    # Filter detections based on confidence threshold
    mask = class_prob > confidence_threshold
    detections = [{
        'box': [x_min[i] - (x_max[i]) / 2,
                y_min[i] - (y_max[i]) / 2,
                x_max[i] + x_min[i] - (x_max[i]) / 2,
                y_max[i] + y_min[i] - (y_max[i]) / 2],
        'class_id': class_id[i],
        'confidence': class_prob[i]
    } for i in range(len(mask)) if mask[i]]

    # Apply non-maximum suppression
    if detections:
        boxes = np.array([detection['box'] for detection in detections])
        confidences = np.array([detection['confidence'] for detection in detections])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), confidence_threshold, nms_threshold)
        if indices.size > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]
        else:
            detections = []
    return detections

# Load mô hình ONNX

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnxruntime.get_device()=='GPU' else ['CPUExecutionProvider']
session = onnxruntime.InferenceSession('weights/best.onnx',providers=["CUDAExecutionProvider"])
cap = cv2.VideoCapture("Tài xế người Trung Quốc ngủ gật khi đang lái xe....mp4")

while True:
    # Bắt đầu đo thời gian
    start_time = time.time()

    _,image = cap.read()
    execution_time1 = time.time() - start_time
    print("Thời gian chụp ảnh: {:.6f} giây".format(execution_time1))
    # image = cv2.imread("image.jpeg")
    image_draw = letterbox(image)
    # Load ảnh đầu vào
    # image = cv2.imread('image.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape


    # Chuyển đổi kích thước và sắp xếp lại kênh màu
    # image_resized = cv2.resize(image, (640, 640))  # Thay đổi kích thước
    image_resized = letterbox(image)
    image_resized = np.transpose(image_resized, (2, 0, 1))  # Sắp xếp lại các kênh màu
    image_resized = np.ascontiguousarray(image_resized)
    # Chạy inference với ONNX Runtime
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if len(image_resized.shape) == 3:
        image_resized = image_resized[None]  # expand for batch dim
    input_data = image_resized.astype(np.float32)
    input_data /= 255
    print(input_data.shape)
    start_time1 = time.time()
    output_data = session.run(None, {input_name: input_data})[0]
    execution_time2 = time.time() - start_time1
    print("Thời gian xử lý: {:.6f} giây".format(execution_time2))


    # Using Make by Me
    results = postprocess_yolov5(output_data)
    print(f"Result:{results}")
    for value in results:
        print(value["box"], value["class_id"], value["confidence"])
        x, y, x_max, y_max = value["box"]
        class_name = names[value["class_id"]]
        cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
        cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                    thickness=2)
  

    # Kết thúc đo thời gian
    end_time = time.time()
    # Tính thời gian thực thi
    execution_time = end_time - start_time
    print("Thời gian thực thi: {:.6f} giây".format(execution_time))


    cv2.imshow("test",image_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
