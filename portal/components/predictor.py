"""Prediction page component for the Streamlit portal."""

import io

import pandas as pd
import streamlit as st

from portal.components.molecule import render_molecule


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
        Results are compared to a reference molecule of your choice.
        """
    )

    # Level 2: Reference molecule selection
    st.subheader("Select Reference Molecule")
    ref_molecules = api_client.get_reference_molecules()

    if not ref_molecules:
        st.warning(
            "Could not load reference molecules. Using default (Cyclopentane). "
            "Make sure the API server is running."
        )
        ref_m, ref_sigma, ref_eps = 2.3655, 3.7114, 288.84
        ref_name = "Cyclopentane"
    else:
        ref_names = ["Custom..."] + [m["name"] for m in ref_molecules]
        selected = st.selectbox(
            "Compare against:",
            ref_names,
            index=ref_names.index("Cyclopentane")
            if "Cyclopentane" in ref_names
            else 1,
        )

        if selected == "Custom...":
            # Level 1: Custom parameter inputs
            with st.expander("Custom reference parameters", expanded=True):
                ref_m = st.number_input("m (segments)", value=2.3655, step=0.01, min_value=0.1)
                ref_sigma = st.number_input(
                    "σ (Å)", value=3.7114, step=0.01, min_value=0.1
                )
                ref_eps = st.number_input(
                    "ε/k (K)", value=288.84, step=1.0, min_value=1.0
                )
                ref_name = "Custom"
        else:
            mol = next(m for m in ref_molecules if m["name"] == selected)
            ref_m, ref_sigma, ref_eps = mol["m"], mol["sigma"], mol["epsilon_k"]
            ref_name = selected
            st.caption(f"SMILES: `{mol['smiles']}` | Source: {mol['source']}")

    st.divider()

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

                    # Applicability domain status as color-coded bar
                    in_domain = pred.get("in_domain", True)
                    tanimoto = pred.get("tanimoto_nn")

                    if tanimoto is not None:
                        if tanimoto >= 0.4 and in_domain:
                            st.success(
                                f"**AD Status: In-Domain** (Tanimoto={tanimoto:.3f})",
                                icon="✅",
                            )
                        elif tanimoto >= 0.3:
                            st.warning(
                                f"**AD Status: Moderate** (Tanimoto={tanimoto:.3f}) — "
                                "treat prediction with caution.",
                                icon="⚠️",
                            )
                        else:
                            st.error(
                                f"**AD Status: Out-of-Domain** (Tanimoto={tanimoto:.3f}) — "
                                "high prediction uncertainty expected.",
                                icon="🚨",
                            )
                    elif not in_domain:
                        st.warning(
                            "**AD Status: Out-of-Domain** — predictions may be unreliable.",
                            icon="⚠️",
                        )

                    # Association warning
                    if pred.get("is_associating", False):
                        st.info(
                            "Molecule has association sites (OH, NH, COOH). "
                            "3-parameter PC-SAFT may be insufficient; "
                            "consider using the association variant.",
                            icon="ℹ️",
                        )

                    # Show predictions with comparison to reference molecule
                    param_names = {
                        "m": "m (segments)",
                        "sigma": "σ (Å)",
                        "epsilon_k": "ε/k (K)",
                    }
                    ref_values = {"m": ref_m, "sigma": ref_sigma, "epsilon_k": ref_eps}

                    # Get uncertainty if available
                    unc = pred.get("uncertainty", {})
                    has_uncertainty = bool(unc)

                    metric_cols = st.columns(3)
                    for idx, param in enumerate(["m", "sigma", "epsilon_k"]):
                        pred_value = pred.get(param)
                        ref_value = ref_values[param]

                        if pred_value is not None:
                            pct_diff = abs(pred_value - ref_value) / ref_value * 100

                            # Format value with uncertainty if available
                            if has_uncertainty:
                                unc_key = f"{param}_std"
                                unc_val = unc.get(unc_key, 0)
                                if param == "epsilon_k":
                                    value_str = f"{pred_value:.2f} ± {unc_val:.2f}"
                                else:
                                    value_str = f"{pred_value:.4f} ± {unc_val:.4f}"
                            else:
                                if param == "epsilon_k":
                                    value_str = f"{pred_value:.2f}"
                                else:
                                    value_str = f"{pred_value:.4f}"

                            # delta_color="inverse" means smaller is better (closer to reference)
                            metric_cols[idx].metric(
                                label=param_names[param],
                                value=value_str,
                                delta=f"{pct_diff:.1f}% diff from {ref_name}",
                                delta_color="inverse",
                            )

                # Similar molecules search
                with st.expander("Find Similar Molecules"):
                    st.markdown(
                        """
                        Search for molecules with similar PC-SAFT parameters or structure
                        from the prediction database or reference library.
                        """
                    )

                    col_k, col_metric, col_corpus = st.columns(3)
                    with col_k:
                        k = st.slider("Number of neighbors", 5, 50, 10, key=f"k_{i}")
                    with col_metric:
                        metric = st.selectbox(
                            "Similarity metric",
                            ["parameter", "tanimoto", "both"],
                            key=f"metric_{i}",
                            help=(
                                "Parameter: distance in PC-SAFT space; "
                                "Tanimoto: fingerprint similarity"
                            ),
                        )
                    with col_corpus:
                        corpus_choice = st.selectbox(
                            "Search corpus",
                            ["all", "reference", "novel"],
                            key=f"corpus_{i}",
                            help="Reference = curated molecules; Novel = model predictions",
                        )

                    if st.button("Find Similar", key=f"similar_{i}"):
                        with st.spinner("Searching..."):
                            response = api_client.find_similar(
                                smiles=pred["smiles"],
                                m=pred.get("m"),
                                sigma=pred.get("sigma"),
                                epsilon_k=pred.get("epsilon_k"),
                                k=k,
                                metric=metric,
                                corpus=corpus_choice,
                            )

                        if "error" in response:
                            st.error(f"Search failed: {response['error']}")
                        else:
                            st.caption(
                                f"Corpus: {response['corpus']} | "
                                f"Version: {response['corpus_version']} | "
                                f"Metric: {response['metric']}"
                            )

                            if not response["neighbors"]:
                                st.warning("No similar molecules found")
                            else:
                                # Format neighbors as DataFrame
                                neighbors_data = []
                                for n in response["neighbors"]:
                                    row = {
                                        "SMILES": n["smiles"],
                                        "Corpus": n["corpus"],
                                        "m": f"{n['m']:.4f}",
                                        "σ (Å)": f"{n['sigma']:.4f}",
                                        "ε/k (K)": f"{n['epsilon_k']:.2f}",
                                    }
                                    if n.get("parameter_distance") is not None:
                                        row["Param Dist"] = f"{n['parameter_distance']:.4f}"
                                    if n.get("tanimoto_similarity") is not None:
                                        row["Tanimoto"] = f"{n['tanimoto_similarity']:.3f}"
                                    if n.get("boiling_point_K") is not None:
                                        row["T_b (K)"] = f"{n['boiling_point_K']:.1f}"
                                    if n.get("mol_class"):
                                        row["Class"] = n["mol_class"]
                                    neighbors_data.append(row)

                                neighbors_df = pd.DataFrame(neighbors_data)
                                st.dataframe(neighbors_df, use_container_width=True)

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
