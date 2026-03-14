"""Data submission page component for the Streamlit portal."""

import streamlit as st
from rdkit import Chem


def render_submission_page(api_client):
    """Render the experimental data submission page.

    Parameters
    ----------
    api_client : PCSAFTClient
        API client for submitting data.
    """
    st.header("Submit Experimental Data")

    st.markdown(
        """
        ### Help Improve the Model

        If you have experimentally measured PC-SAFT parameters, please contribute them here.
        Your data will be validated and incorporated into the next training cycle, improving
        predictions for everyone.

        **Why this matters:**
        - More training data → better predictions
        - Your molecules may be in underrepresented chemical space
        - The system automatically retrains when enough new data accumulates
        """
    )

    with st.form("submission_form"):
        smiles = st.text_input(
            "SMILES",
            placeholder="C1CCCC1",
            help="Molecular SMILES string (canonical or isomeric)",
        )

        col1, col2, col3 = st.columns(3)

        m = col1.number_input(
            "m (segments)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.01,
            help="Number of segments (dimensionless)",
        )

        sigma = col2.number_input(
            "σ (Å)",
            min_value=2.0,
            max_value=6.0,
            value=3.5,
            step=0.01,
            help="Segment diameter in Angstroms",
        )

        epsilon_k = col3.number_input(
            "ε/k (K)",
            min_value=50.0,
            max_value=600.0,
            value=250.0,
            step=0.1,
            help="Dispersion energy in Kelvin",
        )

        source = st.text_input(
            "Source (DOI or lab ID)",
            placeholder="10.1021/je900753e or Lab Notebook 2024-03-15",
            help="Publication DOI or internal lab identifier for traceability",
        )

        submitted = st.form_submit_button("Submit Data", type="primary")

        if submitted:
            # Validation
            errors = []

            if not smiles.strip():
                errors.append("SMILES is required")
            elif Chem.MolFromSmiles(smiles) is None:
                errors.append("Invalid SMILES string")

            if not source.strip():
                errors.append("Source (DOI or lab ID) is required")

            # Parameter range checks (already enforced by number_input, but double-check)
            if not (0.5 <= m <= 10.0):
                errors.append("m must be between 0.5 and 10.0")
            if not (2.0 <= sigma <= 6.0):
                errors.append("σ must be between 2.0 and 6.0 Å")
            if not (50.0 <= epsilon_k <= 600.0):
                errors.append("ε/k must be between 50.0 and 600.0 K")

            if errors:
                for err in errors:
                    st.error(err)
                return

            # Submit to API
            with st.spinner("Submitting..."):
                result = api_client.submit_data(smiles, m, sigma, epsilon_k, source)

            if result.get("status") == "accepted":
                st.success(
                    "✅ Data submitted successfully! "
                    "It will be included in the next training cycle."
                )
                st.balloons()
            else:
                st.error(
                    f"❌ Submission rejected: {result.get('detail', 'Unknown error')}"
                )
                st.info(
                    "Make sure the FastAPI server is running: `uvicorn serving.app:app`"
                )
