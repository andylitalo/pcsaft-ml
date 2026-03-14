"""Prediction page component for the Streamlit portal."""

import io

import pandas as pd
import streamlit as st

from portal.components.molecule import render_molecule

# Cyclopentane reference values from literature
CYCLOPENTANE_REF = {"m": 2.3655, "sigma": 3.7114, "epsilon_k": 288.84}


def render_prediction_page(api_client):
    """Render the prediction input/output page.

    Parameters
    ----------
    api_client : PCSAFTClient
        API client for making predictions.
    """
    st.header("Predict PC-SAFT Parameters")

    st.markdown(
        """
        Enter molecular SMILES strings to predict PC-SAFT equation-of-state parameters.
        Results are compared to **cyclopentane** (a common blowing agent reference).
        """
    )

    # Input methods: text area or CSV upload
    input_method = st.radio("Input method", ["Enter SMILES", "Upload CSV"], horizontal=True)

    smiles_list = []

    if input_method == "Enter SMILES":
        smiles_input = st.text_area(
            "Enter SMILES (one per line)",
            value="C1CCCC1\nCC(F)=CF",
            height=100,
        )
        if smiles_input.strip():
            smiles_list = [s.strip() for s in smiles_input.strip().split("\n") if s.strip()]

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV with a 'smiles' column",
            type=["csv"],
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if "smiles" not in df.columns:
                    st.error("CSV must have a 'smiles' column")
                else:
                    smiles_list = df["smiles"].dropna().astype(str).tolist()
                    st.success(f"Loaded {len(smiles_list)} molecules from CSV")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if st.button("Predict", type="primary", disabled=len(smiles_list) == 0):
        with st.spinner("Predicting..."):
            result = api_client.predict(smiles_list)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
            st.info("Make sure the FastAPI server is running: `uvicorn serving.app:app`")
            return

        predictions = result.get("predictions", [])
        if not predictions:
            st.warning("No predictions returned")
            return

        model_name = result.get("model_name", "unknown")
        st.success(f"Predicted {len(predictions)} molecules using model: {model_name}")

        # Display results
        for i, pred in enumerate(predictions):
            if not pred.get("valid", False):
                st.error(f"Invalid SMILES: {pred.get('smiles', '???')}")
                continue

            with st.container():
                st.divider()
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Render molecule structure
                    try:
                        img = render_molecule(pred["smiles"], size=(250, 250))
                        st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Rendering error: {e}")

                with col2:
                    st.markdown(f"**SMILES**: `{pred['smiles']}`")

                    # Applicability domain warning
                    if not pred.get("in_domain", True):
                        st.warning("⚠️ Molecule may be outside model's applicability domain")

                    # Association warning
                    if pred.get("is_associating", False):
                        st.info(
                            "ℹ️ Molecule has association sites (OH, NH, COOH). "
                            "3-parameter PC-SAFT may be insufficient; "
                            "consider using the association variant."
                        )

                    # Show predictions with comparison to cyclopentane
                    param_names = {
                        "m": "m (segments)",
                        "sigma": "σ (Å)",
                        "epsilon_k": "ε/k (K)",
                    }

                    metric_cols = st.columns(3)
                    for idx, param in enumerate(["m", "sigma", "epsilon_k"]):
                        pred_value = pred.get(param)
                        ref_value = CYCLOPENTANE_REF[param]

                        if pred_value is not None:
                            pct_diff = abs(pred_value - ref_value) / ref_value * 100
                            # delta_color="inverse" means smaller is better (closer to reference)
                            metric_cols[idx].metric(
                                label=param_names[param],
                                value=f"{pred_value:.4f}",
                                delta=f"{pct_diff:.1f}% diff from C5H10",
                                delta_color="inverse",
                            )

                    # Uncertainty estimates if available
                    if pred.get("uncertainty"):
                        with st.expander("Uncertainty estimates"):
                            unc = pred["uncertainty"]
                            st.write(f"m ± {unc.get('m_std', 0):.4f}")
                            st.write(f"σ ± {unc.get('sigma_std', 0):.4f}")
                            st.write(f"ε/k ± {unc.get('epsilon_k_std', 0):.2f}")

        # Offer CSV download of results
        if predictions:
            results_df = pd.DataFrame(
                [
                    {
                        "smiles": p["smiles"],
                        "m": p.get("m"),
                        "sigma": p.get("sigma"),
                        "epsilon_k": p.get("epsilon_k"),
                        "in_domain": p.get("in_domain", True),
                        "is_associating": p.get("is_associating", False),
                        "valid": p.get("valid", False),
                    }
                    for p in predictions
                ]
            )
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="pcsaft_predictions.csv",
                mime="text/csv",
            )
