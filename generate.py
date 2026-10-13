import argparse
import csv
import gc
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from diffusers import DiffusionPipeline


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

MODELS = {
    "sd15": {
        "id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "steps": 20,
        "guidance": 7.5,
    },
    "sd21": {
        "id": "sd2-community/stable-diffusion-2-1-base",
        "steps": 20,
        "guidance": 7.5,
    },
    "turbo": {
        "id": "stabilityai/sd-turbo",
        "steps": 4,
        "guidance": 0.0,
    },
    "ssd1b": {
        "id": "segmind/SSD-1B",
        "steps": 20,
        "guidance": 7.5,
    },
}

PROMPTS = {
    "l_bracket": (
        "Mechanical engineering technical drawing of an L-shaped "
        "metal mounting bracket with a 90 degree bend and two "
        "circular mounting holes. Four separate views arranged "
        "neatly on a white engineering drawing sheet: front view, "
        "top view, side view, and isometric view. Black precise "
        "CAD linework, consistent plate thickness, orthographic "
        "projection, dimension lines and arrows, clean white "
        "background, no shadows, no decorative elements."
    ),
    "shaft_collar": (
        "Mechanical engineering drawing of a cylindrical shaft "
        "collar with a central through bore and one radial set "
        "screw hole. Front, top, side orthographic views and an "
        "isometric view arranged on a white sheet. Black CAD "
        "linework, concentric circles, consistent geometry, "
        "dimension lines and arrows, no shadows."
    ),
    "mounting_flange": (
        "Mechanical engineering drawing of a circular mounting "
        "flange with a central through bore and four equally spaced "
        "bolt holes. Show front, top, side orthographic views and "
        "one isometric view. Black technical CAD lines on white, "
        "concentric geometry, symmetric hole placement, dimension "
        "lines and arrows, no shadows."
    ),
    "clevis": (
        "Mechanical engineering drawing of a clevis bracket with "
        "two parallel prongs, aligned circular holes through the "
        "prongs, and a connecting base. Show front, top, side "
        "orthographic views and an isometric view. Black technical "
        "linework on white, consistent geometry, dimension lines "
        "and arrows, no shadows."
    ),
    "pulley": (
        "Mechanical engineering drawing of a single groove belt "
        "pulley with a central through bore and a circumferential "
        "belt groove. Show front, top, side orthographic views and "
        "one isometric view. Black CAD linework on white, concentric "
        "geometry, dimension lines and arrows, no shadows."
    ),
}

NEGATIVE_PROMPT = (
    "photograph, colored background, textured paper, shadows, "
    "unrelated objects, blurry lines, distorted geometry, "
    "decorative typography, sketch, painting"
)


def monitor_peak_memory(stop_event, peak):
    """Sample process RSS to estimate peak resident memory."""
    process = psutil.Process(os.getpid())

    while not stop_event.is_set():
        try:
            rss = process.memory_info().rss / (1024 ** 2)
            peak[0] = max(peak[0], rss)
        except psutil.Error:
            pass

        stop_event.wait(0.25)


def main():
    parser = argparse.ArgumentParser(
        description="Generate mechanical engineering drawings."
    )

    parser.add_argument(
        "--model", choices=MODELS.keys(), required=True
    )
    parser.add_argument(
        "--component", choices=PROMPTS.keys(), default="l_bracket"
    )
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    if args.size < 256 or args.size % 8 != 0:
        parser.error("--size must be at least 256 and divisible by 8")

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)

    config = MODELS[args.model]
    steps = args.steps if args.steps is not None else config["steps"]
    guidance = config["guidance"]

    print("=" * 60, flush=True)
    print("ENGINEERING DRAWING GENERATION", flush=True)
    print("Model:", args.model, flush=True)
    print("Checkpoint:", config["id"], flush=True)
    print("Component:", args.component, flush=True)
    print("Seed:", args.seed, flush=True)
    print(f"Resolution: {args.size} x {args.size}", flush=True)
    print("Inference steps:", steps, flush=True)
    print("Device: CPU", flush=True)
    print("=" * 60, flush=True)

    start_load = time.perf_counter()

    print("Loading model. This can take several minutes.", flush=True)

    pipe = DiffusionPipeline.from_pretrained(
        config["id"],
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    pipe = pipe.to("cpu")

    load_seconds = time.perf_counter() - start_load
    print(f"Model loaded in {load_seconds:.1f} seconds.", flush=True)

    # Sample process memory during generation.
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / (1024 ** 2)
    peak = [rss_before]
    stop_event = threading.Event()

    monitor = threading.Thread(
        target=monitor_peak_memory,
        args=(stop_event, peak),
        daemon=True,
    )
    monitor.start()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    prompt = PROMPTS[args.component]
    start_generation = time.perf_counter()

    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=args.size,
                height=args.size,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        image = result.images[0]

    finally:
        generation_seconds = time.perf_counter() - start_generation
        stop_event.set()
        monitor.join(timeout=2)

    rss_after = process.memory_info().rss / (1024 ** 2)

    image_name = (
        f"{args.model}_{args.component}_seed{args.seed}.png"
    )
    image_path = RESULTS / image_name
    image.save(image_path)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_alias": args.model,
        "model_id": config["id"],
        "component": args.component,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": args.seed,
        "width": args.size,
        "height": args.size,
        "steps": steps,
        "guidance_scale": guidance,
        "device": "cpu",
        "dtype": "float32",
        "torch_version": torch.__version__,
        "diffusers_version": __import__(
            "diffusers"
        ).__version__,
        "model_load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "rss_before_generation_mb": round(rss_before, 1),
        "rss_after_generation_mb": round(rss_after, 1),
        "peak_process_rss_mb": round(peak[0], 1),
        "image_path": str(image_path.relative_to(ROOT)),
    }

    metadata_path = LOGS / "generation_metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metadata) + "\n")

    csv_path = LOGS / "generation_summary.csv"
    write_header = (
        not csv_path.exists() or csv_path.stat().st_size == 0
    )

    with csv_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(metadata.keys())
        )

        if write_header:
            writer.writeheader()

        writer.writerow(metadata)

    print("\nGeneration completed.", flush=True)
    print("Image:", image_path, flush=True)
    print(f"Loading time: {load_seconds:.1f} seconds", flush=True)
    print(
        f"Generation time: {generation_seconds:.1f} seconds",
        flush=True,
    )
    print(
        f"Peak sampled process memory: {peak[0]:.1f} MB",
        flush=True,
    )
    print("Metadata:", metadata_path, flush=True)
    print("Summary:", csv_path, flush=True)

    del result, image, pipe, generator
    gc.collect()


if __name__ == "__main__":
    main()
