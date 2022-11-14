from diffusers import DiffusionPipeline
from pytorch_lightning import seed_everything
from contextlib import nullcontext
import multiprocessing

import torch
from torch import autocast
import torchvision

import wandb
from torchvision.utils import make_grid
import numpy as np


# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed_everything(42)

prompts = [
    "A photo of an astronaut riding a horse on mars",
    "Baby seal lounging on a beach",
    "Hyperrealistic mesmerizing portrait of Jimi Hendrix floating in spirals of iridescent light",
]
prompt = prompts[0]
N_STEPS = 25
N_SAMPLES = 8
FLASH_ATTENTION = True


def do_inference(pipe, n_samples, use_autocast, num_inference_steps=N_STEPS):
    torch.cuda.empty_cache()
    context = autocast if (device.type == "cuda" and use_autocast) else nullcontext
    with context("cuda"):
        images = pipe(
            prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps
        ).images

    return images


def get_inference_time(
    pipe, n_samples=N_SAMPLES, n_repeats=3, use_autocast=False, num_inference_steps=N_STEPS
):
    from torch.utils.benchmark import Timer

    timer = Timer(
        stmt="do_inference(pipe, n_samples, use_autocast, num_inference_steps)",
        setup="from __main__ import do_inference",
        globals={
            "pipe": pipe,
            "n_samples": n_samples,
            "use_autocast": use_autocast,
            "num_inference_steps": num_inference_steps,
        },
        num_threads=multiprocessing.cpu_count(),
    )
    profile_result = timer.timeit(
        n_repeats
    )  # benchmark.Timer performs 2 iterations for warmup
    return round(profile_result.mean, 2)


def get_inference_memory(pipe, n_samples=N_SAMPLES, use_autocast=False, num_inference_steps=N_STEPS):
    if not torch.cuda.is_available():
        return 0

    torch.cuda.empty_cache()
    context = autocast if (device.type == "cuda" and use_autocast) else nullcontext
    with context("cuda"):
        images = pipe(
            prompt=[prompt] * n_samples, num_inference_steps=num_inference_steps
        ).images

    mem = torch.cuda.memory_reserved()
    return round(mem / 1e9, 2)


def run_experiment(model_id, model_path, size=256):

        run = wandb.init(entity='lambdalabs', project="MiniSD", name=f"{model_id}_{size}")
        pipe = DiffusionPipeline.from_pretrained(model_path).to("cuda")
        def null_safety(images, **kwargs):
            return images, False
        pipe.safety_checker = null_safety

        if FLASH_ATTENTION:
            pipe.enable_xformers_memory_efficient_attention()


        inference_time = get_inference_time(pipe)
        memory_usage = get_inference_memory(pipe)

        wandb.log(
           {"runtime": inference_time,
           "memory": memory_usage})

        for prompt in prompts:
            image = pipe(prompt=prompt, height=size, width=size).images[0]
            wandb.log({ prompt : wandb.Image(image, caption=prompt)})

        run.finish()


def run_experiments():

    # 512x512 run (sd1.4 only)
    run_experiment("sd-v1.4", "CompVis/stable-diffusion-v1-4", 512)

    # 256x256 runs
    for model_path, model_id in [
        ("/home/eole/Desktop/minisd_attention-only_ema", "miniSD_attention-only_ema"),
        (
            "/home/eole/Desktop/minisd_attention-only_no-ema",
            "miniSD_attention-only_no-ema",
        ),
        ("/home/eole/Desktop/minisd_full_ema", "miniSD_full_ema"),
        ("/home/eole/Desktop/minisd_full_no-ema", "miniSD_full_no-ema"),
        ("CompVis/stable-diffusion-v1-4", "sd-v1.4"),
    ]:
        run_experiment(model_id, model_path, size=256)




if __name__ == "__main__":

    run_experiments()
