"""Headless demo driver for the torch-2.x port (no gradio, no mm-stack).

Generates multi-view street images from the bundled demo/data/*.pth samples.
Mirrors interactive_gui.py's load_model_from/run_pipe, minus the GUI.

    python demo/headless.py -m pretrained/MagicDrive-424x800-450ep \
        -d "demo/data/*.pth" -o output/headless
"""
import argparse
import glob
import os
import sys

import torch
from omegaconf import OmegaConf
from hydra import compose, initialize
from functools import partial

sys.path.append(".")  # noqa
from diffusers import UniPCMultistepScheduler
from magicdrive.misc.common import load_module
from magicdrive.runner.img_utils import concat_6_views
from demo.helper import preprocess_fn, precompute_cam_ext


def load_model_from(dir, weight_dtype=torch.float16, device="cuda"):
    original_overrides = OmegaConf.load(os.path.join(dir, "hydra/overrides.yaml"))
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="test_config", overrides=original_overrides)

    pipe_param = {}
    model_cls = load_module(cfg.model.model_module)
    controlnet = model_cls.from_pretrained(
        os.path.join(dir, cfg.model.controlnet_dir), torch_dtype=weight_dtype)
    controlnet.eval()
    pipe_param["controlnet"] = controlnet

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet = unet_cls.from_pretrained(
            os.path.join(dir, cfg.model.unet_dir), torch_dtype=weight_dtype)
        unet.eval()
        pipe_param["unet"] = unet

    pipe_cls = load_module(cfg.model.pipe_module)
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=weight_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    # torch-2.x port: no xformers; PyTorch SDPA is the fast path
    return cfg, pipe


@torch.no_grad()
def run_pipe(cfg, pipe, data, seed=None, prompt="", step=None, scale=None):
    assert cfg.model.bbox_mode == "all-xyz"
    assert cfg.model.bbox_view_shared == False
    preprocess = partial(preprocess_fn, tokenizer=pipe.tokenizer,
                         template=cfg.dataset.template)
    val_input = preprocess(data)

    generator = torch.manual_seed(cfg.seed if seed is None else seed)
    weight_dtype = pipe.unet.dtype
    pipeline_param = {**cfg.runner.pipeline_param}
    if step is not None:
        pipeline_param["num_inference_steps"] = step
    if scale is not None:
        pipeline_param["guidance_scale"] = scale
    image = pipe(
        prompt=[prompt] if prompt else val_input["captions"],
        negative_prompt=None,
        image=val_input["bev_map_with_aux"],
        camera_param=val_input["camera_param"].to(weight_dtype),
        height=cfg.dataset.image_size[0],
        width=cfg.dataset.image_size[1],
        generator=generator,
        bev_controlnet_kwargs=val_input["kwargs"],
        **pipeline_param,
    )
    return concat_6_views(image.images[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="pretrained/MagicDrive-424x800-450ep")
    ap.add_argument("-d", "--data", default="demo/data/*.pth")
    ap.add_argument("-o", "--out", default="output/headless")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()

    cfg, pipe = load_model_from(args.model)
    pipe.set_progress_bar_config(disable=True)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(args.data))[: args.limit]
    for f in files:
        data = torch.load(f, weights_only=False)
        precompute_cam_ext(data)
        img = run_pipe(cfg, pipe, data, seed=args.seed, step=args.steps)
        name = os.path.splitext(os.path.basename(f))[0]
        img.save(os.path.join(args.out, f"{name}.png"))
        print(f"OK {name} -> {args.out}/{name}.png", flush=True)


if __name__ == "__main__":
    main()
