import re

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def parse_composite_metrics(composite_str):
    """Parse the composite string to extract all metrics"""
    if pd.isna(composite_str) or composite_str == "":
        return {}

    metrics = {}
    # Pattern to match metric name and its stats
    pattern = r"(\w+)\s*:\s*avg=\s*([\d.]+),\s*std=\s*([\d.]+),\s*min=\s*([\d.]+),\s*max=\s*([\d.]+)"

    matches = re.findall(pattern, composite_str)
    for match in matches:
        metric_name, avg, std, min_val, max_val = match
        metrics[f"{metric_name}_avg"] = float(avg)
        metrics[f"{metric_name}_std"] = float(std)
        metrics[f"{metric_name}_min"] = float(min_val)
        metrics[f"{metric_name}_max"] = float(max_val)

    return metrics


def parse_sca_value(sca_str):
    """Parse SCA value from format like '75.9115 +/- 5.4467 %'"""
    if pd.isna(sca_str) or sca_str == "":
        return None, None

    match = re.match(r"([\d.]+)\s*\+/-\s*([\d.]+)", str(sca_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def parse_collision_value(collision_str):
    """Parse collision value and IoU from format like '1.6481481481481481, average IoU is 0.3800 %'"""
    if pd.isna(collision_str) or collision_str == "":
        return None, None

    parts = str(collision_str).split(",")
    collision_count = float(parts[0].strip()) if len(parts) > 0 else None

    iou = None
    if len(parts) > 1:
        iou_match = re.search(r"([\d.]+)\s*%", parts[1])
        if iou_match:
            iou = float(iou_match.group(1))

    return collision_count, iou


def parse_stool_value(stool_str):
    """Parse stool count and additional info from format like '14' or '159(3.38 stools per scene on avg)'"""
    if pd.isna(stool_str) or stool_str == "":
        return None, None

    stool_str = str(stool_str).strip()

    # Extract main number
    main_match = re.match(r"(\d+)", stool_str)
    stool_count = int(main_match.group(1)) if main_match else None

    # Extract average stools per scene if present
    avg_match = re.search(r"\(([\d.]+)\s*stools per scene", stool_str)
    avg_stools = float(avg_match.group(1)) if avg_match else None

    return stool_count, avg_stools


# Read the raw CSV file
print("Reading CSV file...")
df_raw = pd.read_csv("experiments_12_oct.csv", sep="|", skipinitialspace=True)

# Clean column names (remove leading/trailing spaces)
df_raw.columns = df_raw.columns.str.strip()

# Remove empty first and last columns if they exist
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]

# Remove rows where experiment desc is just dashes (separator rows)
df_raw = df_raw[~df_raw["experiment desc"].str.contains("---", na=False)]

# Remove completely empty rows
df_raw = df_raw.dropna(how="all")

print(f"Loaded {len(df_raw)} rows")

# Create new dataframe with parsed values
parsed_data = []

for idx, row in df_raw.iterrows():
    record = {}

    # Basic fields
    record["experiment_desc"] = str(row.get("experiment desc", "")).strip()
    record["solver"] = str(row.get("solver ddim 150", "")).strip()
    record["ema"] = str(row.get("ema", "")).strip()
    record["wandb_run_id"] = str(row.get("wandb run id", "")).strip()

    # Numeric fields
    record["kid"] = pd.to_numeric(row.get("kid", np.nan), errors="coerce")
    record["fid"] = pd.to_numeric(row.get("fid", np.nan), errors="coerce")
    record["obj_count"] = pd.to_numeric(row.get("obj count", np.nan), errors="coerce")
    record["ckl"] = pd.to_numeric(row.get("ckl", np.nan), errors="coerce")

    # Parse SCA
    sca_mean, sca_std = parse_sca_value(row.get("sca", ""))
    record["sca_mean"] = sca_mean
    record["sca_std"] = sca_std

    # Parse collision
    collision_count, collision_iou = parse_collision_value(row.get("collision", ""))
    record["collision_count"] = collision_count
    record["collision_avg_iou"] = collision_iou

    # Parse stool
    stool_count, avg_stools_per_scene = parse_stool_value(
        row.get("stool(number of scenes with stool out of 162)", "")
    )
    record["stool_scene_count"] = stool_count
    record["avg_stools_per_scene"] = avg_stools_per_scene

    # Parse composite metrics
    composite_metrics = parse_composite_metrics(row.get("composite", ""))
    record.update(composite_metrics)

    # Add pkl path
    record["pkl_path"] = str(row.get("pkl", "")).strip()

    parsed_data.append(record)

# Create structured dataframe
df_parsed = pd.DataFrame(parsed_data)

# Save to CSV
output_csv = "experiments_12_oct_parsed.csv"
df_parsed.to_csv(output_csv, index=False)
print(f"\n✓ Saved parsed data to: {output_csv}")
print(f"  Total records: {len(df_parsed)}")
print(f"  Total columns: {len(df_parsed.columns)}")

# Display column names
print("\nColumns in parsed CSV:")
for i, col in enumerate(df_parsed.columns, 1):
    print(f"  {i:2d}. {col}")

# Display first few rows
print("\nFirst few rows preview:")
print(df_parsed.head().to_string())

# ============================================================================
# ANALYSIS SECTION
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE ANALYSIS")
print("=" * 80)

# Filter out rows with missing experiment descriptions
df_analysis = df_parsed[df_parsed["experiment_desc"].str.len() > 0].copy()

print(f"\nAnalyzing {len(df_analysis)} experiments...")

# 1. Basic Statistics by Experiment Type
print("\n" + "-" * 80)
print("1. KEY METRICS BY EXPERIMENT TYPE")
print("-" * 80)

metrics_of_interest = [
    "fid",
    "kid",
    "sca_mean",
    "collision_count",
    "obj_count",
    "stool_scene_count",
]

for metric in metrics_of_interest:
    if metric in df_analysis.columns:
        print(f"\n{metric.upper()}:")
        grouped = df_analysis.groupby("experiment_desc")[metric].agg(
            ["mean", "std", "min", "max", "count"]
        )
        print(grouped.to_string())

# 2. EMA Impact Analysis
print("\n" + "-" * 80)
print("2. EMA IMPACT ANALYSIS (TRUE vs FALSE)")
print("-" * 80)

ema_comparison = (
    df_analysis[df_analysis["ema"].isin(["true", "false"])]
    .groupby("ema")[["fid", "kid", "sca_mean", "collision_count"]]
    .agg(["mean", "std"])
)

print(ema_comparison.to_string())

# 3. Solver Type Analysis
print("\n" + "-" * 80)
print("3. SOLVER TYPE ANALYSIS")
print("-" * 80)

solver_analysis = (
    df_analysis[df_analysis["solver"].str.len() > 0]
    .groupby("solver")[["fid", "kid", "sca_mean", "collision_count"]]
    .agg(["mean", "std", "count"])
)

print(solver_analysis.to_string())

# 4. Composite Metrics Analysis
print("\n" + "-" * 80)
print("4. COMPOSITE METRICS ANALYSIS")
print("-" * 80)

composite_cols = [
    col
    for col in df_analysis.columns
    if any(
        metric in col
        for metric in [
            "gravity",
            "object_count",
            "must_have_furniture",
            "non_penetration",
        ]
    )
    and col.endswith("_avg")
]

if composite_cols:
    print("\nAverage values across all experiments:")
    for col in composite_cols:
        mean_val = df_analysis[col].mean()
        std_val = df_analysis[col].std()
        print(f"  {col:30s}: {mean_val:.4f} ± {std_val:.4f}")

# 5. Correlation Analysis
print("\n" + "-" * 80)
print("5. CORRELATION ANALYSIS")
print("-" * 80)

numeric_cols = [
    "fid",
    "kid",
    "sca_mean",
    "collision_count",
    "obj_count",
    "gravity_avg",
    "object_count_avg",
    "non_penetration_avg",
]

available_cols = [col for col in numeric_cols if col in df_analysis.columns]
correlation_matrix = df_analysis[available_cols].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.to_string())

# Find strongest correlations
print("\nStrongest Correlations (|r| > 0.5):")
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            print(
                f"  {correlation_matrix.columns[i]:25s} <-> {correlation_matrix.columns[j]:25s}: {corr_val:+.3f}"
            )

# 6. Stool Analysis
print("\n" + "-" * 80)
print("6. STOOL PRESENCE ANALYSIS")
print("-" * 80)

stool_data = df_analysis[df_analysis["stool_scene_count"].notna()]
print(f"\nTotal experiments with stool data: {len(stool_data)}")
print(f"Average stool scene count: {stool_data['stool_scene_count'].mean():.2f}")
print(f"Max stool scene count: {stool_data['stool_scene_count'].max():.0f}")
print(f"Min stool scene count: {stool_data['stool_scene_count'].min():.0f}")

print("\nStool count by experiment:")
stool_by_exp = df_analysis.groupby("experiment_desc")["stool_scene_count"].agg(
    ["mean", "min", "max"]
)
print(stool_by_exp.to_string())

# 7. Best Performing Configurations
print("\n" + "-" * 80)
print("7. BEST PERFORMING CONFIGURATIONS")
print("-" * 80)

# Lower FID is better
best_fid = df_analysis.loc[df_analysis["fid"].idxmin()]
print(f"\nBest FID Score: {best_fid['fid']:.2f}")
print(f"  Experiment: {best_fid['experiment_desc']}")
print(f"  Solver: {best_fid['solver']}")
print(f"  EMA: {best_fid['ema']}")

# Higher SCA is better
best_sca = df_analysis.loc[df_analysis["sca_mean"].idxmax()]
print(f"\nBest SCA Score: {best_sca['sca_mean']:.2f}")
print(f"  Experiment: {best_sca['experiment_desc']}")
print(f"  Solver: {best_sca['solver']}")
print(f"  EMA: {best_sca['ema']}")

# Lower collision is better
best_collision = df_analysis.loc[df_analysis["collision_count"].idxmin()]
print(f"\nBest Collision Score: {best_collision['collision_count']:.4f}")
print(f"  Experiment: {best_collision['experiment_desc']}")
print(f"  Solver: {best_collision['solver']}")
print(f"  EMA: {best_collision['ema']}")

# Higher non-penetration is better
if "non_penetration_avg" in df_analysis.columns:
    best_non_pen = df_analysis.loc[df_analysis["non_penetration_avg"].idxmax()]
    print(f"\nBest Non-Penetration Score: {best_non_pen['non_penetration_avg']:.4f}")
    print(f"  Experiment: {best_non_pen['experiment_desc']}")
    print(f"  Solver: {best_non_pen['solver']}")
    print(f"  EMA: {best_non_pen['ema']}")

# 8. Trade-off Analysis
print("\n" + "-" * 80)
print("8. TRADE-OFF ANALYSIS")
print("-" * 80)

print("\nFID vs Collision Trade-off:")
print("(Looking for low FID AND low collision)")
df_analysis["fid_collision_product"] = (
    df_analysis["fid"] * df_analysis["collision_count"]
)
best_tradeoff = df_analysis.loc[df_analysis["fid_collision_product"].idxmin()]
print(f"  Best: {best_tradeoff['experiment_desc']}")
print(
    f"  FID: {best_tradeoff['fid']:.2f}, Collision: {best_tradeoff['collision_count']:.4f}"
)

# 9. Summary Statistics
print("\n" + "-" * 80)
print("9. OVERALL SUMMARY STATISTICS")
print("-" * 80)

summary_stats = df_analysis[
    ["fid", "kid", "sca_mean", "collision_count", "obj_count"]
].describe()
print(summary_stats.to_string())

# 10. Experiment Type Summary
print("\n" + "-" * 80)
print("10. EXPERIMENT TYPE SUMMARY")
print("-" * 80)

exp_counts = df_analysis["experiment_desc"].value_counts()
print(f"\nNumber of variants per experiment type:")
print(exp_counts.to_string())

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Create visualizations
print("\nGenerating visualizations...")

# Create output directory for plots
output_dir = Path("analysis_plots")
output_dir.mkdir(exist_ok=True)

# Plot 1: FID by Experiment Type
fig, ax = plt.subplots(figsize=(14, 6))
df_plot = df_analysis[df_analysis["experiment_desc"].str.len() > 0].copy()
exp_fid = df_plot.groupby("experiment_desc")["fid"].mean().sort_values()
exp_fid.plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("FID Score (lower is better)")
ax.set_title("Average FID Score by Experiment Type")
plt.tight_layout()
plt.savefig(output_dir / "fid_by_experiment.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_dir / 'fid_by_experiment.png'}")

# Plot 2: Collision by Experiment Type
fig, ax = plt.subplots(figsize=(14, 6))
exp_collision = (
    df_plot.groupby("experiment_desc")["collision_count"].mean().sort_values()
)
exp_collision.plot(kind="barh", ax=ax, color="coral")
ax.set_xlabel("Collision Count (lower is better)")
ax.set_title("Average Collision Count by Experiment Type")
plt.tight_layout()
plt.savefig(output_dir / "collision_by_experiment.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_dir / 'collision_by_experiment.png'}")

# Plot 3: Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlation Heatmap of Key Metrics")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_dir / 'correlation_heatmap.png'}")

# Plot 4: FID vs Collision Scatter
fig, ax = plt.subplots(figsize=(10, 8))
scatter_data = df_analysis[df_analysis["experiment_desc"].str.len() > 0].copy()
for exp_type in scatter_data["experiment_desc"].unique():
    exp_data = scatter_data[scatter_data["experiment_desc"] == exp_type]
    ax.scatter(
        exp_data["fid"], exp_data["collision_count"], label=exp_type, alpha=0.6, s=100
    )
ax.set_xlabel("FID Score")
ax.set_ylabel("Collision Count")
ax.set_title("FID vs Collision Trade-off")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "fid_vs_collision.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_dir / 'fid_vs_collision.png'}")

# Plot 5: Composite Metrics Comparison
if composite_cols:
    fig, ax = plt.subplots(figsize=(12, 6))
    composite_means = df_analysis[composite_cols].mean().sort_values()
    composite_means.plot(kind="barh", ax=ax, color="mediumseagreen")
    ax.set_xlabel("Average Score")
    ax.set_title("Average Composite Metric Scores Across All Experiments")
    ax.set_xlim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / "composite_metrics.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: {output_dir / 'composite_metrics.png'}")

# Plot 6: Stool Count Distribution
fig, ax = plt.subplots(figsize=(12, 6))
stool_by_exp_plot = (
    df_analysis.groupby("experiment_desc")["stool_scene_count"].mean().sort_values()
)
stool_by_exp_plot.plot(kind="barh", ax=ax, color="purple")
ax.set_xlabel("Average Stool Scene Count (out of 162)")
ax.set_title("Stool Presence by Experiment Type")
plt.tight_layout()
plt.savefig(output_dir / "stool_by_experiment.png", dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {output_dir / 'stool_by_experiment.png'}")

print("\n✓ All visualizations saved to 'analysis_plots/' directory")
print("\n" + "=" * 80)
print("SCRIPT EXECUTION COMPLETE")
print("=" * 80)
