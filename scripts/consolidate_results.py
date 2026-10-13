
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "results" / "analysis"
out.mkdir(parents=True, exist_ok=True)

generation = pd.read_csv(root / "logs/generation_summary.csv")
scores = pd.read_csv(out / "automated_image_scores.csv")

keys = ["component", "seed"]
merged = generation.merge(
    scores,
    left_on=["model_alias", *keys],
    right_on=["model", *keys],
    how="inner",
    validate="one_to_one",
)

if len(merged) != 60:
    raise ValueError(f"Expected 60 matched images, got {len(merged)}")

# Use each image's score for its own target component.
merged["component_feature_score"] = merged.apply(
    lambda r: r[f"features_{r['component']}_relative_score"], axis=1
)

quality_cols = [
    "technical_drawing_relative_score",
    "orthographic_views_relative_score",
    "dimension_annotations_relative_score",
    "line_clarity_relative_score",
]

merged["visual_proxy_score"] = merged[quality_cols].mean(axis=1)
merged["combined_proxy_score"] = (
    merged["visual_proxy_score"] + merged["component_feature_score"]
) / 2

merged.to_csv(out / "consolidated_image_results.csv", index=False)

summary = merged.groupby("model_alias").agg(
    images=("image_path", "count"),
    generation_mean_s=("generation_seconds", "mean"),
    generation_std_s=("generation_seconds", "std"),
    generation_median_s=("generation_seconds", "median"),
    model_load_mean_s=("model_load_seconds", "mean"),
    peak_rss_mean_mb=("peak_process_rss_mb", "mean"),
    peak_rss_max_mb=("peak_process_rss_mb", "max"),
    technical_drawing=("technical_drawing_relative_score", "mean"),
    orthographic_views=("orthographic_views_relative_score", "mean"),
    dimension_annotations=("dimension_annotations_relative_score", "mean"),
    line_clarity=("line_clarity_relative_score", "mean"),
    component_feature=("component_feature_score", "mean"),
    combined_proxy=("combined_proxy_score", "mean"),
).reset_index().rename(columns={"model_alias": "model"})

summary["generation_mean_min"] = summary["generation_mean_s"] / 60
summary["peak_rss_mean_gb"] = summary["peak_rss_mean_mb"] / 1024
summary["peak_rss_max_gb"] = summary["peak_rss_max_mb"] / 1024
summary = summary.sort_values("combined_proxy", ascending=False)
summary.to_csv(out / "model_comparison_summary.csv", index=False)

by_component = merged.groupby(["model_alias", "component"]).agg(
    images=("image_path", "count"),
    generation_mean_s=("generation_seconds", "mean"),
    technical_drawing=("technical_drawing_relative_score", "mean"),
    orthographic_views=("orthographic_views_relative_score", "mean"),
    dimension_annotations=("dimension_annotations_relative_score", "mean"),
    line_clarity=("line_clarity_relative_score", "mean"),
    component_feature=("component_feature_score", "mean"),
    combined_proxy=("combined_proxy_score", "mean"),
).reset_index().rename(columns={"model_alias": "model"})
by_component.to_csv(out / "model_component_comparison.csv", index=False)

# Plot mean generation time.
time_data = summary.sort_values("generation_mean_s")
ax = time_data.plot(
    x="model", y="generation_mean_s", kind="bar", legend=False,
    title="Mean Generation Time by Model", ylabel="Seconds per image"
)
plt.tight_layout()
plt.savefig(out / "generation_time_comparison.png", dpi=180)
plt.close()

# Plot quality proxies.
summary.set_index("model")[
    ["technical_drawing", "orthographic_views",
     "dimension_annotations", "line_clarity", "component_feature"]
].plot(
    kind="bar", title="Automated Image-Text Similarity Proxies",
    ylabel="Mean relative CLIP score", ylim=(0, 1)
)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(out / "quality_proxy_comparison.png", dpi=180)
plt.close()

# Plot speed versus combined proxy score.
fig, ax = plt.subplots()
ax.scatter(summary["generation_mean_s"], summary["combined_proxy"])
for _, row in summary.iterrows():
    ax.annotate(
        row["model"],
        (row["generation_mean_s"], row["combined_proxy"])
    )
ax.set_xlabel("Mean generation time (seconds per image)")
ax.set_ylabel("Combined automated proxy score")
ax.set_title("Speed versus Automated Quality Proxy")
plt.tight_layout()
plt.savefig(out / "speed_vs_quality.png", dpi=180)
plt.close()

print(f"Matched images: {len(merged)}")
print("\nModel comparison:")
print(summary[[
    "model", "images", "generation_mean_s", "generation_std_s",
    "peak_rss_mean_gb", "peak_rss_max_gb",
    "technical_drawing", "orthographic_views",
    "dimension_annotations", "line_clarity",
    "component_feature", "combined_proxy"
]].round(3).to_string(index=False))
print(f"\nResults saved to: {out}")

