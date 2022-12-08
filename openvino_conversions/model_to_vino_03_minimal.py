from pathlib import Path
# Works with older 2021 version:
import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore
ie = IECore()

import sys
sys.path.append('../')
sys.path.append('../ravaen_payload/')


import logging
import numpy as np
import glob
from typing import Callable
import time

class OpenVinoModel:
    def __init__(self, ie, model_path):
        self.logger = logging.getLogger("model")
        print('\tModel path: {}'.format(model_path))
        self.net = ie.read_network(model_path, model_path[:-4] + '.bin')
        self.set_batch_size(64)

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in self.net.input_info:
            new_shape = [batch] + self.net.input_info[input_layer].input_data.shape[1:]
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)


def device_available():
    ie = IECore()
    devs = ie.available_devices
    return 'MYRIAD' in devs


def load_model(model_path, device='MYRIAD') -> Callable:
    # Broadly copied from the OpenVINO Python examples
    ie = IECore()
    try:
        ie.unregister_plugin('MYRIAD')
    except:
        pass

    model = OpenVinoModel(ie, model_path)
    tic = time.time()
    print('Loading ONNX network to ',device,'...')

    exec_net = ie.load_network(network=model.net, device_name=device,config=None, num_requests=1)
    toc = time.time()
    print('one, time elapsed : {} seconds'.format(toc - tic))

    def predict(x: np.ndarray) -> np.ndarray:
        """
        Predict function using the myriad chip

        Args:
            x: (C, H, W) 3d tensor

        Returns:
            (C, H, W) 3D network logits

        """
        result = exec_net.infer({'input': x[np.newaxis]})
        return result['output'][0]  # (n_class, H, W)

    return predict

BATCH_SIZE = 8
example_input = np.random.rand(BATCH_SIZE, 4, 32, 32)

#from model_to_vino_03_minimal import *
model_path = "../weights_openvino/encoder_model.onnx"
device = 'CPU'
ie = IECore()
model = OpenVinoModel(ie, model_path)
exec_net = ie.load_network(network=model.net, device_name=device, config=None, num_requests=1)

print("Debug inputs:", list(exec_net.inputs.keys()), "and outputs:", list(exec_net.outputs.keys()))

num_images = 100
start = time.perf_counter()
for _ in range(num_images):
    result = exec_net.infer({'input.1': example_input})
    example_output = result['36']
end = time.perf_counter()
time_onnx = end - start
print(
    f"ONNX model in OpenVINO CPU Runtime/CPU: {time_onnx/num_images:.3f} "
    f"seconds per image, FPS: {num_images/time_onnx:.2f}"
)
print("example_input", example_input.shape)
print("example_output", example_output.shape)
res_onnx_cpu = example_output

### ON MYRIAD
res_onnx_myriad = None
try:
    print("trying on MYRIAD!")
    model_path = "../weights_openvino/encoder_model.onnx"
    device = 'MYRIAD'
    ie = IECore()
    model = OpenVinoModel(ie, model_path)
    exec_net = ie.load_network(network=model.net, device_name=device, config=None, num_requests=1)

    print("Debug inputs:", list(exec_net.inputs.keys()), "and outputs:", list(exec_net.outputs.keys()))

    num_images = 100
    start = time.perf_counter()
    for _ in range(num_images):
        result = exec_net.infer({'input.1': example_input})
        example_output = result['36']
    end = time.perf_counter()
    time_onnx = end - start
    print(
        f"ONNX model in OpenVINO MYRIAD Runtime/CPU: {time_onnx/num_images:.3f} "
        f"seconds per image, FPS: {num_images/time_onnx:.2f}"
    )
    print("example_input", example_input.shape)
    print("example_output", example_output.shape)
    res_onnx_myriad = example_output
except Exception as e:
    print("failed with", e)

### Compare with the torch output

# 3. Versus torch model
from model_to_vino_01_convert import EncoderOnly
import torch

TILE_SIZE = 32
keep_latent_log_var = True
model_encoder = EncoderOnly(keep_latent_log_var=keep_latent_log_var)
model_path = "../weights/model_rgbnir"
model_encoder.encoder.load_state_dict(torch.load(model_path + "_encoder.pt"))
model_encoder.fc_mu.load_state_dict(torch.load(model_path + "_fc_mu.pt"))
if keep_latent_log_var:
    model_encoder.fc_var.load_state_dict(torch.load(model_path + "_fc_var.pt"))
model = model_encoder.cpu().eval()
print("Loaded model")


with torch.no_grad():
    start = time.perf_counter()
    for _ in range(num_images):
        result_torch,_ = model(torch.as_tensor(example_input).float())
    end = time.perf_counter()
    time_torch = end - start
print(
    f"PyTorch model on CPU: {time_torch/num_images:.3f} seconds per image, "
    f"FPS: {num_images/time_torch:.2f}"
)

print("result_torch", result_torch.shape)


#### Equals?

print("res_onnx_cpu", res_onnx_cpu.shape, res_onnx_cpu.dtype)
print("result_torch", result_torch.shape, result_torch.dtype)
if res_onnx_myriad is not None:
    print("res_onnx_myriad", res_onnx_myriad.shape, res_onnx_myriad.dtype)

k = 10
print(res_onnx_cpu[0][0:k])
print(result_torch[0][0:k])
if res_onnx_myriad is not None:
    print(res_onnx_myriad[0][0:k])

