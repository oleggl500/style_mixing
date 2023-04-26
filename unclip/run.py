import argparse
from diffusers import DiffusionPipeline
import torch
import math
import PIL
from PIL import Image

def unclip_image_interpolation(
  start_image,
  end_image,
  steps,
  seed,
  pipe
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    images = [start_image, end_image]
    output = pipe(image=images, steps=steps, generator=generator)
    return output.images
    


def run(content,style,pipe):
    style = Image.open(style)
    content = Image.open(content)
    res = unclip_image_interpolation(style,content,10,44,pipe)
    res[6].save("./remix.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("content")
    parser.add_argument("style")
    print("Remixing started")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.bfloat16

    pipe = DiffusionPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=dtype, custom_pipeline='unclip_image_interpolation')
    pipe.to(device)

    run(args.content,args.style,pipe)
