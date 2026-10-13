
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
OUT = RESULTS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cpu"

COMPONENT_FEATURES = {
    "l_bracket": (
        "an L shaped metal bracket with a right angle bend "
        "and two circular mounting holes",
        "a cylindrical mechanical part without an L shaped bracket",
    ),
    "shaft_collar": (
        "a cylindrical shaft collar with a central bore "
        "and a radial set screw hole",
        "a flat rectangular plate without a cylindrical bore",
    ),
    "mounting_flange": (
        "a circular mounting flange with a central bore "
        "and four equally spaced bolt holes",
        "an irregular object without a central bore or bolt holes",
    ),
    "clevis": (
        "a clevis bracket with two parallel prongs, aligned holes "
        "and a connecting base",
        "a solid cylinder without parallel prongs or aligned holes",
    ),
    "pulley": (
        "a belt pulley with a central bore and circumferential groove",
        "a flat plate without a central bore or circumferential groove",
    ),
}

CRITERIA = {
    "technical_drawing": (
        "a clean black and white mechanical engineering line drawing",
        "a photorealistic colored 3D render of a mechanical object",
    ),
    "orthographic_views": (
        "a technical drawing showing consistent front top and side "
        "orthographic views of the same mechanical component",
        "a single perspective view of a mechanical object",
    ),
    "dimension_annotations": (
        "an engineering drawing with readable dimension lines arrows "
        "and measurement annotations",
        "an engineering drawing without dimension lines or annotations",
    ),
    "line_clarity": (
        "a clean precise high contrast technical drawing with sharp "
        "black lines on a white background",
        "a blurry low contrast drawing with messy indistinct lines",
    ),
}


def component_from_filename(filename):
    match = re.match(
        r"^(sd15|sd21|turbo|ssd1b)_(.+)_seed(\d+)\.png$",
        filename,
    )
    if not match:
        return None
    return match.groups()


def edge_density(image):
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    array = np.asarray(edges, dtype=np.uint8)
    # Fraction of pixels with strong local intensity changes.
    return float(np.mean(array > 40))


def main():
    torch.set_num_threads(4)

    print("Loading CLIP model on CPU. First run downloads model files.")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()

    # Each criterion uses a positive and a negative text description.
    text_pairs = {}
    for name, pair in CRITERIA.items():
        text_pairs[name] = pair

    for name, pair in COMPONENT_FEATURES.items():
        text_pairs[f"features_{name}"] = pair

    all_texts = []
    pair_positions = {}

    for name, pair in text_pairs.items():
        pair_positions[name] = (len(all_texts), len(all_texts) + 1)
        all_texts.extend(pair)

    tokens = processor(
        text=all_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        text_features = model.get_text_features(**tokens)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    records = []
    image_paths = sorted(RESULTS.glob("*.png"))

    for index, path in enumerate(image_paths, start=1):
        parsed = component_from_filename(path.name)
        if parsed is None:
            continue

        model_name, component, seed = parsed

        if component not in COMPONENT_FEATURES:
            continue

        print(f"[{index}] Evaluating {path.name}", flush=True)

        with Image.open(path) as source:
            image = source.convert("RGB")
            edge_score = edge_density(image)

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

        with torch.inference_mode():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

        # Cosine similarities between this image and all descriptions.
        similarities = (
            image_features @ text_features.T
        ).squeeze(0).cpu().numpy()

        record = {
            "image_id": path.stem,
            "model": model_name,
            "component": component,
            "seed": int(seed),
            "edge_density": edge_score,
        }

        for criterion, (positive_idx, negative_idx) in pair_positions.items():
            pos = float(similarities[positive_idx])
            neg = float(similarities[negative_idx])

            # Relative preference between two descriptions.
            # This is not a probability of correctness.
            shifted = np.array([pos, neg]) * 10.0
            shifted -= shifted.max()
            probs = np.exp(shifted) / np.exp(shifted).sum()

            record[criterion + "_similarity"] = pos
            record[criterion + "_relative_score"] = float(probs[0])

        records.append(record)

    if not records:
        raise RuntimeError(
            "No experiment images found. Check the filenames in results/."
        )

    df = pd.DataFrame(records)
    df.to_csv(OUT / "automated_image_scores.csv", index=False)

    metric_cols = [
        col for col in df.columns
        if col.endswith("_relative_score")
    ]

    summary = (
        df.groupby("model")[metric_cols + ["edge_density"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "automated_model_summary.csv")

    component_summary = (
        df.groupby(["model", "component"])[
            metric_cols + ["edge_density"]
        ]
        .mean()
        .reset_index()
    )
    component_summary.to_csv(
        OUT / "automated_component_summary.csv",
        index=False,
    )

    # One chart per evaluation criterion.
    for metric in metric_cols:
        means = df.groupby("model")[metric].mean().sort_values()

        ax = means.plot(
            kind="bar",
            title=metric.replace("_relative_score", "").replace("_", " ").title(),
            ylabel="Relative CLIP preference (0 to 1)",
        )
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUT / f"{metric}.png", dpi=180)
        plt.close()

    means = df.groupby("model")["edge_density"].mean().sort_values()
    ax = means.plot(
        kind="bar",
        title="Mean edge density",
        ylabel="Fraction of strong edge pixels",
    )
    plt.tight_layout()
    plt.savefig(OUT / "edge_density.png", dpi=180)
    plt.close()

    print("\nEvaluation complete.")
    print("Images evaluated:", len(df))
    print("Models:", sorted(df["model"].unique()))
    print("Results:", OUT)
    print("\nMean relative scores by model:")
    print(df.groupby("model")[metric_cols].mean().round(3))


if __name__ == "__main__":
    main()

