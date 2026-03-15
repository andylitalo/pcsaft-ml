"""Example usage of the pcsaft-predict package.

This script demonstrates the core functionality of the package.
Run with: python examples.py
"""

import pcsaft_predict


def example_single_prediction():
    """Example 1: Predict parameters for a single molecule."""
    print("=" * 60)
    print("Example 1: Single Molecule Prediction")
    print("=" * 60)

    smiles = "CCO"  # ethanol
    df = pcsaft_predict.predict(smiles)

    print(f"\nPredicting PC-SAFT parameters for {smiles} (ethanol):")
    print(df.to_string(index=False))
    print()


def example_batch_prediction():
    """Example 2: Predict parameters for multiple molecules."""
    print("=" * 60)
    print("Example 2: Batch Prediction")
    print("=" * 60)

    smiles_list = [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CCCCC",  # n-pentane
        "C1CCCC1",  # cyclopentane
    ]

    df = pcsaft_predict.predict(smiles_list)

    print("\nPredicting PC-SAFT parameters for 4 molecules:")
    print(df[["smiles", "m", "sigma", "epsilon_k", "in_domain"]].to_string(index=False))
    print()


def example_uncertainty():
    """Example 3: Predict with uncertainty quantification."""
    print("=" * 60)
    print("Example 3: Uncertainty Quantification")
    print("=" * 60)

    smiles_list = ["CCO", "c1ccccc1", "CCCCC"]
    df = pcsaft_predict.predict_with_uncertainty(smiles_list)

    print("\nPredictions with uncertainty estimates:")
    print(
        df[["smiles", "m", "m_std", "sigma", "sigma_std", "epsilon_k", "epsilon_k_std"]]
        .to_string(index=False)
    )
    print()


def example_applicability_domain():
    """Example 4: Filter by applicability domain."""
    print("=" * 60)
    print("Example 4: Applicability Domain Filtering")
    print("=" * 60)

    # Include some unusual molecules
    smiles_list = [
        "CCO",  # ethanol (common)
        "C1CCCC1",  # cyclopentane (common)
        "FC(F)(F)C(F)(F)F",  # perfluoroethane (less common)
        "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # large PAH (out of domain)
    ]

    df = pcsaft_predict.predict(smiles_list)

    print("\nMolecules with AD scores:")
    print(df[["smiles", "in_domain", "tanimoto_nn"]].to_string(index=False))

    print("\nFiltering to in-domain molecules only:")
    in_domain = df[df["in_domain"]]
    print(in_domain[["smiles", "m", "sigma", "epsilon_k"]].to_string(index=False))
    print()


def example_list_models():
    """Example 5: List available models."""
    print("=" * 60)
    print("Example 5: Available Models")
    print("=" * 60)

    models = pcsaft_predict.list_models()
    print(f"\nAvailable models: {models}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("pcsaft-predict Package Examples")
    print("=" * 60 + "\n")

    example_single_prediction()
    example_batch_prediction()
    example_uncertainty()
    example_applicability_domain()
    example_list_models()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
