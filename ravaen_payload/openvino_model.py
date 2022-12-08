# Source: https://github.com/gonzmg88/wildride_unibap_dorbit/blob/main/worldfloods_payload/myriad_model.py

from pathlib import Path
# Works with older 2021 version:
import openvino
import openvino.inference_engine
from openvino.inference_engine import IECore
ie = IECore()

import logging
import numpy as np
import glob
from typing import Callable
import time
import torch

class OpenVinoModel:
    def __init__(self, ie, model_path, batch_size=64):
        self.logger = logging.getLogger("model")
        print('\tModel path: {}'.format(model_path))
        self.net = ie.read_network(model_path, model_path[:-4] + '.bin')
        self.set_batch_size(batch_size)

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

def get_prediction_function(model_path="encoder_model.onnx", device='MYRIAD', batch_size=64):
    #device = 'CPU'
    #device = 'MYRIAD'
    ie = IECore()
    try:
        ie.unregister_plugin('MYRIAD')
    except:
        pass

    model = OpenVinoModel(ie, model_path, batch_size)
    exec_net = ie.load_network(network=model.net, device_name=device, config=None, num_requests=1)

    print("Debug inputs:", list(exec_net.inputs.keys()), "and outputs:", list(exec_net.outputs.keys()))

    example_input = np.random.rand(batch_size, 4, 32, 32)
    result = exec_net.infer({'input.1': example_input})
    example_output = result['36']

    # print("example_input", example_input.shape)
    # print("example_output", example_output.shape)

    def predict(x: np.ndarray) -> np.ndarray:
        result = exec_net.infer({'input.1': x[np.newaxis]})
        return result['36'] # just the mus

    return predict



def encode_batch_openvino(openvino_predictionfunc, xs, to_and_from_torch = True):
    #openvino_predictionfunc = get_prediction_function(model_path="encoder_model.onnx", device='MYRIAD', batch_size=64)
    mus = openvino_predictionfunc(xs)
    return mus

