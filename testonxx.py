import time
import numpy as np
import onnxruntime
import cv2

names = {
  0: "normal",
  1: "drowsy",
  2: "drowsy#2",
  3: "yawning",
 }
def postprocess_yolov5(output_data, confidence_threshold=0.5, nms_threshold=0.6):
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

    # for i in range(grid_size):
    #
    #     x_min, y_min, x_max, y_max, confidence = output_data[0, i, :5]
    #
    #     class_probs = output_data[0, i, 5:]
    #     class_id = np.argmax(class_probs)
    #
    #     class_prob = class_probs[class_id] * confidence
    #
    #     # print(x_min, y_min, x_max, y_max, confidence, np.argmax(class_probs))
    #
    #     # Check confidence threshold
    #     if class_prob > confidence_threshold:
    #         # Calculate bounding box coordinates relative to the grid cell
    #         width = x_max
    #         height = y_max
    #         # Append detection to list
    #         detections.append({
    #             'box': [x_min - width /2 , y_min - height / 2 , x_max + (x_min - width /2) ,  y_max + (y_min - height / 2)],
    #             'class_id': class_id,
    #             'confidence': class_prob
    #         })

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
# def process_yolo_output(output_data, num_classes, confidence_threshold=0.7):
#     grid_size = output_data.shape[1]
#     for i in range(grid_size):
#         conf_class = output_data[0,i,5:]
#         x_min,y_min,x_max,y_max,conf = output_data[0,i,:5]
#         class_id = np.argmax(conf_class)
#         if conf > confidence_threshold:
#             print(x_min,y_min,x_max,y_max,conf,class_id)



# Load mô hình ONNX

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnxruntime.get_device()=='GPU' else ['CPUExecutionProvider']
print(providers)
session = onnxruntime.InferenceSession('weights/best.onnx')
cap = cv2.VideoCapture(0)

while True:
    # Bắt đầu đo thời gian
    start_time = time.time()

    _,image = cap.read()
    execution_time1 = time.time() - start_time
    print("Thời gian chụp ảnh: {:.6f} giây".format(execution_time1))
    # image = cv2.imread("image.jpeg")
    image_draw = cv2.resize(image,(640,640))
    # Load ảnh đầu vào
    # image = cv2.imread('image.jpeg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image = image / 255.0


    # Chuyển đổi kích thước và sắp xếp lại kênh màu
    image_resized = cv2.resize(image, (640, 640))  # Thay đổi kích thước
    image_resized = np.transpose(image_resized, (2, 0, 1))  # Sắp xếp lại các kênh màu

    # Chạy inference với ONNX Runtime
    input_name = session.get_inputs()[0].name
    print(input_name)
    output_name = session.get_outputs()[0].name
    input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

    start_time1 = time.time()
    output_data = session.run([output_name], {input_name: input_data})[0]
    execution_time2 = time.time() - start_time1
    print("Thời gian xử lý: {:.6f} giây".format(execution_time2))
    # output_data = session.run(None, {'images': input_data_float32})[0]
    # output_data = np.transpose(output_data,(0,2,1))

    # # using Non_Max_Suppression
    # output= torch.from_numpy(output_data)
    # out = non_max_suppression(output, conf_thres=0.5, iou_thres=0.5)
    # # Tạo một list để chứa các numpy arrays từ các tensors trong out
    # numpy_arrays = []
    # # Chuyển mỗi tensor thành numpy array và thêm vào list numpy_arrays
    # for tensor in out:
    #     numpy_arrays.append(tensor.numpy())
    #
    # # Ghép các numpy arrays lại thành một numpy array lớn
    # numpy_array_out = np.concatenate(numpy_arrays, axis=0)

    # In ra numpy array

    # Using Make by Me
    results = postprocess_yolov5(output_data)
    for value in results:
        print(value["box"], value["class_id"], value["confidence"])
        x, y, x_max, y_max = value["box"]
        class_name = names[value["class_id"]]
        cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
        cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                    thickness=2)
    # try:
    #     for result in numpy_array_out:
    #         x, y, x_max, y_max,score,cls_id = result
    #         class_name = names[cls_id]
    #         cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
    #         cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
    #                     thickness=2)
    # except:
    #     pass

    # Kết thúc đo thời gian
    end_time = time.time()
    # Tính thời gian thực thi
    execution_time = end_time - start_time
    print("Thời gian thực thi: {:.6f} giây".format(execution_time))


    cv2.imshow("test",image_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # # In ra các thông tin của các đối tượng được dự đoán
    # for i in range(len(boxes)):
    #     print(f"Object {i+1}: Class ID {class_ids[i]}, Confidence {confidences[i]}, Bounding Box {boxes[i]}")