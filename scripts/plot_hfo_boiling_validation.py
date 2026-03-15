"""Generate boiling point validation parity plot for Step 24."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from screening.hfo_screening import validate_boiling_points, validate_boiling_points_rf

# Run validations
df_lit = validate_boiling_points()
df_rf = validate_boiling_points_rf()

# Filter to only valid predictions
df_lit_valid = df_lit[~df_lit["T_b_predicted_K"].isna()].copy()
df_rf_valid = df_rf[~df_rf["T_b_predicted_K"].isna()].copy()

print(f"Literature PC-SAFT params: {len(df_lit_valid)}/{len(df_lit)} valid")
print(f"RF-predicted PC-SAFT params: {len(df_rf_valid)}/{len(df_rf)} valid")

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Literature parameters
if len(df_lit_valid) > 0:
    ax1.scatter(
        df_lit_valid["T_b_experimental_K"],
        df_lit_valid["T_b_predicted_K"],
        s=100,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add compound names as labels
    for _, row in df_lit_valid.iterrows():
        name_short = row["name"].split("(")[0].strip() if row["name"] else row["smiles"][:10]
        ax1.annotate(
            name_short,
            (row["T_b_experimental_K"], row["T_b_predicted_K"]),
            fontsize=10,
            ha="right",
            va="bottom",
            alpha=0.7,
        )

    # Diagonal line
    min_T = min(
        df_lit_valid["T_b_experimental_K"].min(),
        df_lit_valid["T_b_predicted_K"].min(),
    )
    max_T = max(
        df_lit_valid["T_b_experimental_K"].max(),
        df_lit_valid["T_b_predicted_K"].max(),
    )
    ax1.plot(
        [min_T, max_T],
        [min_T, max_T],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Perfect prediction",
    )

    # Compute metrics
    mae = df_lit_valid["error_K"].abs().mean()
    rmse = np.sqrt((df_lit_valid["error_K"] ** 2).mean())
    y_pred = df_lit_valid["T_b_predicted_K"]
    y_true = df_lit_valid["T_b_experimental_K"]
    ss_res = ((y_pred - y_true) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    textstr = f"n = {len(df_lit_valid)}\nMAE = {mae:.1f} K\nRMSE = {rmse:.1f} K\nR² = {r2:.3f}"
    ax1.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
else:
    ax1.text(
        0.5,
        0.5,
        "No valid predictions\n(VLE solver failed)",
        transform=ax1.transAxes,
        fontsize=16,
        ha="center",
        va="center",
    )

ax1.set_xlabel("Experimental T_b (K)", fontsize=16)
ax1.set_ylabel("Predicted T_b (K)", fontsize=16)
ax1.set_title("Boiling Point Validation\n(Literature PC-SAFT Parameters)", fontsize=18)
ax1.tick_params(labelsize=14)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.3)

# Plot 2: RF-predicted parameters
if len(df_rf_valid) > 0:
    ax2.scatter(
        df_rf_valid["T_b_experimental_K"],
        df_rf_valid["T_b_predicted_K"],
        s=100,
        alpha=0.7,
        color="orange",
        edgecolor="black",
        linewidth=1,
    )

    # Add SMILES as labels (no names in RF predictions)
    for _, row in df_rf_valid.iterrows():
        label = row["smiles"][:12] + "..." if len(row["smiles"]) > 12 else row["smiles"]
        ax2.annotate(
            label,
            (row["T_b_experimental_K"], row["T_b_predicted_K"]),
            fontsize=9,
            ha="right",
            va="bottom",
            alpha=0.7,
        )

    # Diagonal line
    min_T = min(
        df_rf_valid["T_b_experimental_K"].min(),
        df_rf_valid["T_b_predicted_K"].min(),
    )
    max_T = max(
        df_rf_valid["T_b_experimental_K"].max(),
        df_rf_valid["T_b_predicted_K"].max(),
    )
    ax2.plot(
        [min_T, max_T],
        [min_T, max_T],
        "k--",
        linewidth=2,
        alpha=0.5,
        label="Perfect prediction",
    )

    # Compute metrics
    mae = df_rf_valid["error_K"].abs().mean()
    rmse = np.sqrt((df_rf_valid["error_K"] ** 2).mean())
    y_pred = df_rf_valid["T_b_predicted_K"]
    y_true = df_rf_valid["T_b_experimental_K"]
    ss_res = ((y_pred - y_true) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)

    textstr = f"n = {len(df_rf_valid)}\nMAE = {mae:.1f} K\nRMSE = {rmse:.1f} K\nR² = {r2:.3f}"
    ax2.text(
        0.05,
        0.95,
        textstr,
        transform=ax2.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
else:
    ax2.text(
        0.5,
        0.5,
        "No valid predictions\n(VLE solver failed)",
        transform=ax2.transAxes,
        fontsize=16,
        ha="center",
        va="center",
    )

ax2.set_xlabel("Experimental T_b (K)", fontsize=16)
ax2.set_ylabel("Predicted T_b (K)", fontsize=16)
ax2.set_title("Boiling Point Validation\n(RF-Predicted PC-SAFT Parameters)", fontsize=18)
ax2.tick_params(labelsize=14)
ax2.legend(fontsize=12)
ax2.grid(alpha=0.3)

plt.tight_layout()

# Save figure
output_path = (
    Path(__file__).parent.parent
    / "figures"
    / "24_hfo_boiling_point_validation"
    / "boiling_point_parity.png"
)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nSaved to: {output_path}")

# Print summary
print("\n=== Literature PC-SAFT Parameters ===")
if len(df_lit_valid) > 0:
    cols = ["smiles", "name", "T_b_experimental_K", "T_b_predicted_K", "error_K"]
    print(df_lit_valid[cols].to_string(index=False))
else:
    print("No valid predictions (VLE solver failed for all compounds)")

print("\n=== RF-Predicted PC-SAFT Parameters ===")
if len(df_rf_valid) > 0:
    cols = ["smiles", "T_b_experimental_K", "T_b_predicted_K", "error_K"]
    print(df_rf_valid[cols].to_string(index=False))
else:
    print("No valid predictions (VLE solver failed for all compounds)")
