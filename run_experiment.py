import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

MODELS = ["sd15", "sd21", "turbo", "ssd1b"]
COMPONENTS = [
    "l_bracket",
    "shaft_collar",
    "mounting_flange",
    "clevis",
    "pulley",
]
SEEDS = [1001, 1002, 1003]

RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

summary_path = LOGS / "batch_experiment.csv"
failures_path = LOGS / "batch_failures.csv"

total = len(MODELS) * len(COMPONENTS) * len(SEEDS)
completed = 0
skipped = 0
failed = 0

with summary_path.open("a", newline="", encoding="utf-8") as summary_file:
    writer = csv.DictWriter(
        summary_file,
        fieldnames=[
            "model", "component", "seed", "status",
            "elapsed_seconds", "return_code", "log_file",
        ],
    )
    if summary_path.stat().st_size == 0:
        writer.writeheader()

    for model in MODELS:
        for component in COMPONENTS:
            for seed in SEEDS:
                image_path = (
                    RESULTS / f"{model}_{component}_seed{seed}.png"
                )

                if image_path.exists():
                    print(f"[SKIP] {image_path.name}", flush=True)
                    skipped += 1
                    continue

                log_path = (
                    LOGS / f"{model}_{component}_seed{seed}.log"
                )
                command = [
                    sys.executable, "-u", str(ROOT / "generate.py"),
                    "--model", model,
                    "--component", component,
                    "--seed", str(seed),
                    "--size", "512",
                ]

                print(
                    f"\n[{completed + skipped + failed + 1}/{total}] "
                    f"{model} | {component} | seed={seed}",
                    flush=True,
                )

                start = time.perf_counter()

                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )

                elapsed = round(time.perf_counter() - start, 2)
                success = result.returncode == 0 and image_path.exists()
                status = "success" if success else "failed"

                writer.writerow({
                    "model": model,
                    "component": component,
                    "seed": seed,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "return_code": result.returncode,
                    "log_file": str(log_path.relative_to(ROOT)),
                })
                summary_file.flush()

                if success:
                    completed += 1
                    print(
                        f"[OK] {image_path.name}, {elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    failed += 1
                    print(
                        f"[FAILED] See {log_path}",
                        flush=True,
                    )

with failures_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "component", "seed", "log_file"])

    for model in MODELS:
        for component in COMPONENTS:
            for seed in SEEDS:
                image_path = (
                    RESULTS / f"{model}_{component}_seed{seed}.png"
                )
                log_path = (
                    LOGS / f"{model}_{component}_seed{seed}.log"
                )

                if log_path.exists() and not image_path.exists():
                    writer.writerow([
                        model, component, seed,
                        str(log_path.relative_to(ROOT)),
                    ])

print("\nBatch experiment finished.")
print(f"Successful new runs: {completed}")
print(f"Skipped existing images: {skipped}")
print(f"Failed runs: {failed}")
print(f"Batch log: {summary_path}")
print(f"Failure list: {failures_path}")
