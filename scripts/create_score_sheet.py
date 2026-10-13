import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "logs" / "human_evaluation.csv"

models = ["sd15", "sd21", "turbo", "ssd1b"]
components = [
    "l_bracket", "shaft_collar", "mounting_flange",
    "clevis", "pulley",
]
seeds = [1001, 1002, 1003]

fields = [
    "image_id",
    "model",
    "component",
    "seed",
    "feature_correctness_0_5",
    "orthographic_consistency_0_5",
    "dimension_annotation_0_5",
    "line_clarity_0_5",
    "prompt_adherence_0_5",
    "notes",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for model in models:
        for component in components:
            for seed in seeds:
                image_id = f"{model}_{component}_seed{seed}"
                image_path = RESULTS / f"{image_id}.png"

                if not image_path.exists():
                    print("Missing:", image_id)
                    continue

                writer.writerow({
                    "image_id": image_id,
                    "model": model,
                    "component": component,
                    "seed": seed,
                    "feature_correctness_0_5": "",
                    "orthographic_consistency_0_5": "",
                    "dimension_annotation_0_5": "",
                    "line_clarity_0_5": "",
                    "prompt_adherence_0_5": "",
                    "notes": "",
                })

print("Created:", OUTPUT)
