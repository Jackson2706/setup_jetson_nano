import torch
from torch.onnx import export

# Load the YOLOv5 PyTorch model
model_path = "weights/best.pt"
model =  torch.hub.load('./yolov5', 'custom', source ='local', path=model_path,force_reload=True)
model.eval()

# Create dummy input (replace this with your real input)
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust the shape according to your YOLOv5 model

# Export the model to ONNX
onnx_filename = "best.onnx"  # Name for the output ONNX file
export(model, dummy_input, onnx_filename, opset_version=12, verbose=True)
