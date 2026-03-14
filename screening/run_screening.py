"""Main CLI orchestrator for the blowing agent screening pipeline.

Usage:
    python -m screening.run_screening [--skip-patents] [--sa-threshold 4.5]
"""

import argparse
from pathlib import Path

import pandas as pd

from screening.filters import (
    filter_patents,
    filter_pcsaft_similarity,
    filter_synthesizability,
)
from screening.generate import generate_hfo_candidates

RESULTS_DIR = Path(__file__).parent / "results"


def run_screening(skip_patents: bool = False, sa_threshold: float = 4.5):
    """Run the full 4-stage screening pipeline.

    Parameters
    ----------
    skip_patents : bool
        If True, skip the PubChem patent check (useful for offline testing).
    sa_threshold : float
        SA Score cutoff for synthesizability filter.
    """
    print("=" * 60)
    print("Blowing Agent Screening Pipeline")
    print("=" * 60)

    # Stage 1: Generate candidates
    print("\n[1/4] Generating HFO candidates...")
    candidates = generate_hfo_candidates()

    # Stage 2: SA Score filter
    print("\n[2/4] Filter 1: Synthetic Accessibility Score...")
    sa_passed = filter_synthesizability(candidates, threshold=sa_threshold)

    # Stage 3: Patent filter
    if skip_patents:
        print("\n[3/4] Filter 2: Patent check — SKIPPED")
        patent_passed = sa_passed
    else:
        print("\n[3/4] Filter 2: PubChem patent check...")
        patent_passed = filter_patents(sa_passed)

    # Stage 4: PC-SAFT similarity
    print("\n[4/4] Filter 3: PC-SAFT parameter prediction + cyclopentane ranking...")
    ranked = filter_pcsaft_similarity(patent_passed)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ranked_candidates.csv"

    if ranked:
        df = pd.DataFrame(ranked)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        print("\nTop 10 candidates (closest to cyclopentane):")
        print(df.head(10).to_string(index=False))
    else:
        print("\nNo candidates passed all filters.")

    print("\n" + "=" * 60)
    print("Screening complete.")
    print("=" * 60)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Run blowing agent screening pipeline")
    parser.add_argument(
        "--skip-patents",
        action="store_true",
        help="Skip PubChem patent check (for offline use)",
    )
    parser.add_argument(
        "--sa-threshold",
        type=float,
        default=4.5,
        help="SA Score threshold (default: 4.5)",
    )
    args = parser.parse_args()
    run_screening(skip_patents=args.skip_patents, sa_threshold=args.sa_threshold)


if __name__ == "__main__":
    main()
