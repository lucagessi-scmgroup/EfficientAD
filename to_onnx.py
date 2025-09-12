import torch
import torchvision.models as models
from common import *

teacher_base = "models/teacher_small.pth"
teacher_weights = "teacher_final.pth"
student_weights = "student_final.pth"

# Load your PyTorch model (e.g., ResNet34)
teacher = get_pdn_small(out_channels=384)
state_dict = torch.load(teacher_base, map_location='cpu')
teacher.load_state_dict(state_dict)
teacher.eval()

student = get_pdn_small(out_channels=384)
state_dict = torch.load(teacher_base, map_location='cpu')
student.load_state_dict(state_dict)
student.eval()

# Dummy input for tracing
dummy_input = torch.randn(1, 3, 32, 128)

# Export the model to ONNX
torch.onnx.export(
    teacher,
    dummy_input,
    "teacher.onnx",
    opset_version=17, # Specify opset version
    input_names=["input"],
    output_names=["output"]
)
torch.onnx.export(
    student,
    dummy_input,
    "student.onnx",
    opset_version=17, # Specify opset version
    input_names=["input"],
    output_names=["output"]
)