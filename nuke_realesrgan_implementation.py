"""File for converting the pretrained Real-ESRGAN model into a torchscript file."""
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from torch import nn


def _get_model_state_dict() -> dict:
    """Get the pretrained state dict.

    Returns:
        State dict for the x4plus training.
    """
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    model_path = load_file_from_url(url=model_url)
    loadnet = torch.load(model_path, map_location=torch.device("cpu"))

    # prefer to use params_ema
    if "params_ema" in loadnet:
        keyname = "params_ema"
    else:
        keyname = "params"
    return loadnet[keyname]


class TiledREALESRGAN(nn.Module):
    """Wrapper to convert the RealESRGAN model to a nuke compatible torchscript file."""

    def __init__(self):
        super().__init__()
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        model.load_state_dict(_get_model_state_dict(), strict=True)
        model.eval()
        self.model = model

    def forward(self, input_tensor: torch.Tensor):
        return self.model(input_tensor)


def main():
    model = TiledREALESRGAN()
    module = torch.jit.script(model)
    module.save("output/realesrgan_tiled.pt")


if __name__ == "__main__":
    main()
