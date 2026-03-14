"""Validate submitted PC-SAFT data."""

from kfp import dsl


def validate_data_logic(
    input_csv: str, valid_csv: str, rejected_csv: str
) -> int:
    """Core logic for validating PC-SAFT submissions.

    Parameters
    ----------
    input_csv : str
        Path to input submissions CSV.
    valid_csv : str
        Path to save validated rows.
    rejected_csv : str
        Path to save rejected rows.

    Returns
    -------
    int
        Number of valid rows.
    """
    import pandas as pd
    from rdkit import Chem

    # Read submissions
    df = pd.read_csv(input_csv)

    # Validate SMILES
    valid_rows = []
    rejected_rows = []

    for idx, row in df.iterrows():
        reject_reason = None
        smiles = row.get("smiles", "")

        # Check SMILES validity
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            reject_reason = "invalid_smiles"

        # Range checks
        m = row.get("m")
        sigma = row.get("sigma")
        epsilon_k = row.get("epsilon_k")

        if reject_reason is None:
            if not (0.5 <= m <= 10):
                reject_reason = "m_out_of_range"
            elif not (2.0 <= sigma <= 6.0):
                reject_reason = "sigma_out_of_range"
            elif not (50 <= epsilon_k <= 600):
                reject_reason = "epsilon_k_out_of_range"

        if reject_reason:
            row_dict = row.to_dict()
            row_dict["reject_reason"] = reject_reason
            rejected_rows.append(row_dict)
        else:
            valid_rows.append(row.to_dict())

    # Save outputs
    valid_df = pd.DataFrame(valid_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    valid_df.to_csv(valid_csv, index=False)
    # Only write rejected CSV if there are rejected rows
    if len(rejected_df) > 0:
        rejected_df.to_csv(rejected_csv, index=False)
    else:
        # Write empty file to ensure output exists
        with open(rejected_csv, "w") as f:
            f.write("")

    return len(valid_df)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.1.0", "rdkit==2023.9.1"],
)
def validate_data(
    input_csv: str,
    valid_csv: dsl.OutputPath(str),
    rejected_csv: dsl.OutputPath(str),
) -> int:
    """Validate submitted PC-SAFT data.

    Checks:
    - SMILES validity via RDKit
    - m in [0.5, 10]
    - sigma in [2.0, 6.0]
    - epsilon_k in [50, 600]

    Parameters
    ----------
    input_csv : str
        Path to submitted PC-SAFT data CSV.
    valid_csv : OutputPath(str)
        Path to save validated rows CSV.
    rejected_csv : OutputPath(str)
        Path to save rejected rows CSV with rejection reason.

    Returns
    -------
    int
        Number of valid rows.
    """
    return validate_data_logic(input_csv, valid_csv, rejected_csv)
