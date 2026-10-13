from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "contact_sheets"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = ["sd15", "sd21", "turbo", "ssd1b"]
COMPONENTS = [
    "l_bracket",
    "shaft_collar",
    "mounting_flange",
    "clevis",
    "pulley",
]
SEEDS = [1001, 1002, 1003]

THUMB = (300, 300)
LABEL_H = 28
GAP = 12


def make_sheet(component, seed):
    canvas = Image.new(
        "RGB",
        (4 * THUMB[0] + 5 * GAP, THUMB[1] + LABEL_H + 2 * GAP),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, model in enumerate(MODELS):
        filename = f"{model}_{component}_seed{seed}.png"
        path = RESULTS / filename
        x = GAP + index * (THUMB[0] + GAP)

        if not path.exists():
            draw.text((x, GAP), f"MISSING: {model}", fill="red")
            continue

        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(THUMB)

        tile = Image.new("RGB", THUMB, "white")
        tile.paste(
            image,
            ((THUMB[0] - image.width) // 2,
             (THUMB[1] - image.height) // 2),
        )

        canvas.paste(tile, (x, GAP))
        draw.text(
            (x, GAP + THUMB[1] + 5),
            model,
            fill="black",
        )

    path = OUTPUT / f"{component}_seed{seed}.png"
    canvas.save(path)
    print(path)


for component in COMPONENTS:
    for seed in SEEDS:
        make_sheet(component, seed)
