import sys
sys.path.append('../')
from pathlib import Path
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from typing import List, Any, Dict, Tuple
from model_functions import DownConv, ResConvBlock
import cv2
import numpy as np
import torch
try:
    # newer version only:
    from openvino.runtime import Core
    ie = Core()
    print("from openvino.runtime import Core < works")
except:
    # try older version:
    import openvino
    import openvino.inference_engine
    from openvino.inference_engine import IECore
    ie = IECore()
    print("from openvino.inference_engine import IECore < works")


model_path = Path("encoder_model").with_suffix(".pth")
onnx_path = model_path.with_suffix(".onnx")
ir_path = model_path.with_suffix(".xml")
example_input = np.random.rand(1,4,32,32)
print(example_input.shape)

# 1. ONNX Model in OpenVINO Runtime
# Load the network in OpenVINO Runtime.
ie = Core()
model_onnx = ie.read_model(model=onnx_path)
compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

output_layer_onnx = compiled_model_onnx.output(0)

# Run inference on the input image.
res_onnx = compiled_model_onnx([example_input])[output_layer_onnx]
print("res_onnx", res_onnx.shape)

# 2. OpenVINO IR Model in OpenVINO Runtime
# Load the network in OpenVINO Runtime.
ie = Core()
model_ir = ie.read_model(model=ir_path)
compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

# Get input and output layers.
output_layer_ir = compiled_model_ir.output(0)

# Run inference on the input image.
res_ir = compiled_model_ir([example_input])[output_layer_ir]
print("res_ir", res_ir.shape)

# 3. Versus torch model
from model_to_vino_01_convert import EncoderOnly

TILE_SIZE = 32
keep_latent_log_var = True
model_encoder = EncoderOnly(keep_latent_log_var=keep_latent_log_var)
model_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/weights/model_rgbnir"
model_encoder.encoder.load_state_dict(torch.load(model_path + "_encoder.pt"))
model_encoder.fc_mu.load_state_dict(torch.load(model_path + "_fc_mu.pt"))
if keep_latent_log_var:
    model_encoder.fc_var.load_state_dict(torch.load(model_path + "_fc_var.pt"))
model = model_encoder.cpu().eval()
print("Loaded model")

with torch.no_grad():
    result_torch,_ = model(torch.as_tensor(example_input).float())
    print("result_torch", result_torch.shape)


### compare?

# print("res_onnx", res_onnx.shape, res_onnx.dtype)
# print("res_ir", res_ir.shape, res_ir.dtype)
# print("result_torch", result_torch.shape, result_torch.dtype)
#
# k = 10
# print(res_onnx[0][0:k])
# print(res_ir[0][0:k])
# print(result_torch[0][0:k])

### Timing:
import time

num_images = 64

start = time.perf_counter()
for _ in range(num_images):
    compiled_model_onnx([example_input])
end = time.perf_counter()
time_onnx = end - start
print(
    f"ONNX model in OpenVINO Runtime/CPU: {time_onnx/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_onnx:.2f}"
)

start = time.perf_counter()
for _ in range(num_images):
    compiled_model_ir([example_input])
end = time.perf_counter()
time_ir = end - start
print(
    f"OpenVINO IR model in OpenVINO Runtime/CPU: {time_ir/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_ir:.2f}"
)

with torch.no_grad():
    start = time.perf_counter()
    for _ in range(num_images):
        model(torch.as_tensor(example_input).float())
    end = time.perf_counter()
    time_torch = end - start
print(
    f"PyTorch model on CPU: {time_torch/num_images:.3f} seconds per image, "
    f"FPS: {num_images/time_torch:.2f}"
)

if "GPU" in ie.available_devices:
    compiled_model_onnx_gpu = ie.compile_model(model=model_onnx, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_onnx_gpu([example_input])
    end = time.perf_counter()
    time_onnx_gpu = end - start
    print(
        f"ONNX model in OpenVINO/GPU: {time_onnx_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_onnx_gpu:.2f}"
    )

    compiled_model_ir_gpu = ie.compile_model(model=model_ir, device_name="GPU")
    start = time.perf_counter()
    for _ in range(num_images):
        compiled_model_ir_gpu([example_input])
    end = time.perf_counter()
    time_ir_gpu = end - start
    print(
        f"IR model in OpenVINO/GPU: {time_ir_gpu/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_ir_gpu:.2f}"
    )
else:
    print("GPU not available")

print("--- available:")
devices = ie.available_devices
for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
