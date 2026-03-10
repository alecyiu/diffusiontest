import torch
from diffusers import DiffusionPipeline

def main():
    print("Hello from diffusiontest!")
    pipe = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128", dtype=torch.bfloat16, device_map="cuda")

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipe(prompt).images[0]


if __name__ == "__main__":
    main()
