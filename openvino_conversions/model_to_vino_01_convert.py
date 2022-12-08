### NOTE:
### Had to use the latest version of openvino for the conversions into ONNX and IR (even though the docker later uses older openvino version)
###  ... later we just load the ONNX

# STEPS:
# - isolate just the Encoder bit (into a simple pytorch model)
import sys
sys.path.append('../')

# from tile_classifier.standalone_encoder import get_model
# model, _ = get_model()
# print(model)


from pathlib import Path

from torch import nn, Tensor
import torch
from torch.nn import functional as F
from typing import List, Any, Dict, Tuple
from model_functions import DownConv, ResConvBlock


class EncoderOnly(nn.Module):
    def __init__(self, input_shape=(4, 32, 32), hidden_channels=[16, 32, 64], latent_dim=128,
                 extra_depth_on_scale=0,visualisation_channels=[0, 1, 2], keep_latent_log_var=False) -> None:
        super().__init__()

        self.visualisation_channels = visualisation_channels
        self.latent_dim = latent_dim

        print("\nINPUT shape:", input_shape)
        print("\nLATENT SPACE size:", latent_dim)

        # Calculate size of encoder output
        encoder_output_width = int(input_shape[1] / (2 ** len(hidden_channels)))
        encoder_output_dim = int(encoder_output_width ** 2 * hidden_channels[-1])
        self.encoder_output_shape = (hidden_channels[-1], encoder_output_width, encoder_output_width)

        in_channels = input_shape[0]

        self.encoder = self._build_encoder([in_channels] + hidden_channels, extra_depth_on_scale)

        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)

        self.keep_latent_log_var = keep_latent_log_var
        if self.keep_latent_log_var:
            self.fc_var = nn.Linear(encoder_output_dim, latent_dim)

    @staticmethod
    def _build_encoder(channels, extra_depth):
        in_channels = channels[0]
        encoder = []
        for out_channels in channels[1:]:
            encoder += [
                DownConv(
                    in_channels,
                    out_channels,
                    activation=nn.LeakyReLU,
                    batchnorm=True
                )
            ]
            if extra_depth > 0:
                encoder += [
                    ResConvBlock(
                        out_channels,
                        out_channels,
                        activation=nn.LeakyReLU,
                        batchnorm=True,
                        depth=extra_depth
                    )
                ]
            # for next time round loop
            in_channels = out_channels

        return nn.Sequential(*encoder)


    def encode(self, input: Tensor, verbose=False) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = self.encoder(torch.nan_to_num(input)) # < nan_to_num not supported by ONNX!

        if verbose:
            x = input
            print("input", x.shape)
            for multilayer in self.encoder:
                for layer in multilayer.conv:
                    x = layer(x)
                    print(layer, " => it's output:\n", x.shape)

        result = torch.flatten(result, start_dim=1)
        if verbose: print("result", result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        if verbose: print("mu", self.fc_mu, " => it's output:\n", mu.shape)

        if not self.keep_latent_log_var:
            # speed up if I don't care about the var
            log_var = None
        else:
            log_var = self.fc_var(result)
        if verbose: print("log_var", self.fc_var, " => it's output:\n", log_var.shape)

        return [mu, log_var]

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        return mu, log_var # NOTE: with this model version, we keep both the mu's and logvars


class EncoderOnly_MeansOnly(EncoderOnly):

    def encode(self, input: Tensor, verbose=False) -> List[Tensor]:
        result = self.encoder(input)
        # result = self.encoder(torch.nan_to_num(input)) # < nan_to_num not supported by ONNX!
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        return mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu = self.encode(input)
        return mu


if __name__ == "__main__":

    TILE_SIZE = 32
    # keep_latent_log_var = False # ONNX isnt happy about Nones
    keep_latent_log_var = True

    # V1: full encoder model
    OUT_NAME = "encoder_model"
    model_encoder = EncoderOnly(keep_latent_log_var=keep_latent_log_var)

    # V2: only the mu's part of the model
    OUT_NAME = "encoder_model_mu"
    model_encoder = EncoderOnly_MeansOnly(keep_latent_log_var=keep_latent_log_var)

    # print(model_encoder)
    model_path = "/home/vitek/Vitek/Work/Trillium_RaVAEn_2/codes/RaVAEn-unibap-dorbit/weights/model_rgbnir"
    # model_path = "weights/model_rgbnir"

    model_encoder.encoder.load_state_dict(torch.load(model_path + "_encoder.pt"))
    model_encoder.fc_mu.load_state_dict(torch.load(model_path + "_fc_mu.pt"))
    if keep_latent_log_var:
        model_encoder.fc_var.load_state_dict(torch.load(model_path + "_fc_var.pt"))

    # example_input = torch.randn(64,4,32,32)
    # print("example_input", example_input.shape)
    #
    # example_output, _ = model_encoder(example_input)
    # print("example_output", example_output.shape)

    model = model_encoder.cpu().eval()
    print("Loaded model")

    # Save the model.
    model_path = Path(OUT_NAME).with_suffix(".pth")
    onnx_path = model_path.with_suffix(".onnx")
    ir_path = model_path.with_suffix(".xml")

    torch.save(model.state_dict(), str(model_path))
    print(f"Model saved at {model_path}")

    # - convert model into OPENVINO
    import cv2
    import numpy as np
    import torch
    ###maybe i dont even need this? ## from openvino.runtime import Core

    if not onnx_path.exists():
        dummy_input = torch.randn(1, 4, TILE_SIZE, TILE_SIZE)

        # For the Fastseg model, setting do_constant_folding to False is required
        # for PyTorch>1.5.1
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=11,
            do_constant_folding=False,
        )
        print(f"ONNX model exported to {onnx_path}.")
    else:
        print(f"ONNX model {onnx_path} already exists.")

    # Now to OPENVINO

    # Construct the command for Model Optimizer.
    mo_command = f"""mo
                     --input_model "{onnx_path}"
                     --input_shape "[1,4, {TILE_SIZE}, {TILE_SIZE}]"
                     --data_type FP16
                     --output_dir "{model_path.parent}"
                     """
    # --mean_values = "[123.675, 116.28 , 103.53]"
    # --scale_values = "[58.395, 57.12 , 57.375]"

    mo_command = " ".join(mo_command.split())
    print("Model Optimizer command to convert the ONNX model to OpenVINO:")
    print(mo_command)

    # using the same environment ... run:
    # mo --input_model "encoder_model.onnx" --input_shape "[1,4, 32, 32]" --data_type FP16 --output_dir "."

    #  mo --input_model "encoder_model_mu.onnx" --input_shape "[1,4, 32, 32]" --data_type FP16 --output_dir .

    # - run on dedicated HW

    # - (write 'run_inference' for the VINO model)


