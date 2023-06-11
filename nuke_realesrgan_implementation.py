"""File for converting the pretrained Real-ESRGAN model into a torchscript file."""
import torch
from modified_rrdbnet_arch import RRDBNet
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

    def __init__(self, tile_size: int = 200):
        """Initialize the Tiled RealESRGAN.

        Args:
            tile_size: size in pixels for the side length of the square tiles.
        """
        super().__init__()
        self.scale = 4
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.scale,
        )
        model.load_state_dict(_get_model_state_dict(), strict=True)
        model.eval()
        self.model = model
        self.tile_size = tile_size
        self.tile_pad = 10
        """Overlap between the individual tiles."""

    def forward(self, input_tensor: torch.Tensor):
        return self.forward_tiled(input_tensor)

    def forward_tiled(self, tensor: torch.Tensor):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Copied from: https://github.com/xinntao/Real-ESRGAN (Real-ESRGAN/realesrgan/utils.py:117)
        """
        batch, channel, height, width = tensor.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = torch.zeros(output_shape)
        tiles_x = torch.ceil(width / self.tile_size)
        tiles_y = torch.ceil(height / self.tile_size)
        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = tensor[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                output_tile = self.model(input_tile)
                print(f"RealESRGAN: Processing tile {tile_idx}/{tiles_x * tiles_y}")

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
        return output


def main():
    model = TiledREALESRGAN()
    module = torch.jit.script(model)
    module.save("output/realesrgan_tiled.pt")


if __name__ == "__main__":
    main()
