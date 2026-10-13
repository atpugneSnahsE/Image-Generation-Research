from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

models = ["sd15", "sd21", "turbo", "ssd1b"]
size = (384, 384)
label_height = 36
margin = 12

canvas = Image.new(
    "RGB",
    (4 * (size[0] + margin) + margin,
     size[1] + label_height + 2 * margin),
    "white",
)
draw = ImageDraw.Draw(canvas)

for i, model in enumerate(models):
    path = RESULTS / f"{model}_l_bracket_seed1001.png"

    if not path.exists():
        print(f"Missing pilot: {path}")
        continue

    image = Image.open(path).convert("RGB")
    image.thumbnail(size)

    tile = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    tile.paste(image, (x, y))

    px = margin + i * (size[0] + margin)
    py = margin
    canvas.paste(tile, (px, py))
    draw.text((px, py + size[1] + 8), model, fill="black")

output = RESULTS / "pilot_comparison.png"
canvas.save(output)
print(f"Saved: {output}")
