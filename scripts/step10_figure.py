"""Generate Step 10 figure: Esper-only baseline (ML-SAFT integration pending)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("figures/10_mlsaft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Results from Phase 1 Esper-only retrain (confirmed identical)
models = ["RF (Esper only)\nPhase 1", "RF (Esper only)\nPhase 2 retrain"]
targets = ["m (segments)", "σ (Å)", "ε/k (K)"]

r2_phase1 = [0.62, 0.35, 0.33]
r2_phase2 = [0.6194, 0.3527, 0.3298]

x = np.arange(len(targets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, r2_phase1, width, label="Phase 1 (original)", color="#2196F3", alpha=0.85)
bars2 = ax.bar(x + width / 2, r2_phase2, width, label="Phase 2 retrain (Esper)", color="#4CAF50", alpha=0.85)

ax.set_xlabel("PC-SAFT Parameter", fontsize=12)
ax.set_ylabel("R² (test set)", fontsize=12)
ax.set_title("Step 10: RF Retrain Baseline\n(ML-SAFT integration pending — Esper only)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.set_ylim(0, 0.85)
ax.legend()
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="R²=0.5 reference")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
out = OUTPUT_DIR / "r2_comparison_esper_baseline.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
