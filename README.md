# RealESRGAN upscaling model for Nuke13.1 (and newer versions)
This is a simple project for converting the trained model from https://github.com/xinntao/Real-ESRGAN
into a nuke compatible cat file for inference. Foundry provides their own [model](https://community.foundry.com/cattery/37767/real-esrgan), but it lacks the support
for low-end GPUs. In my tests, I was only able to upscale an 1K image to a 4K image on a GPU with 8GB of 
Vram. The converted model of this project is a little less memory hungry and allows to you to use any input
size for upscaling. This comes of course with one big downside: it's a lot slower than the original implementation.
IMHO it's still worth to give it a shot because with the implementation of this project, you can upscale almost every input 
even on a GTX1070 GPU.

# Using it in Nuke
Just download the `realesrgan_tiled_v2.cat` from the output folder and load it in the inference node of nuke 13.
It's that easy. If you want to do it 100% correct, you first need to convert the image into srgb color space. Or use the 
gizmo from foundry and replace their cat file with this one.

# Converting the model on your own

Unfortunately I can't manage to set up a `requirements.txt` that can install torch version 1.6.0 on their own.
I already tried to use `--find-links` or `--extra-index-url` in the file but none of them worked.
If you have any idea, please leave a comment :D

## Anyway: how to set up your project:
Run the following two commands after each other:

```
pip install basicsr
pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
The second command will install the required torch version for nuke 13. 
It will not be completely compatible with `basicsr` but it will be enough to pass
the torchscript conversion.

If you want to use it in nuke 14 only, you can skip the installation of the torch version 1.6.0


# Comparison: Tiled model vs. Original model
TODO