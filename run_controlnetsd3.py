from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
import torch

controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                          controlnet=controlnet)

# Move pipeline to CUDA
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
control_image = load_image("docs/source/en/imgs/diffusers_library.jpg")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)

image = pipe(prompt, num_inference_steps=20, generator=generator, control_image=control_image).images[0]

image.save("./output.png")
