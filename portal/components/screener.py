"""Batch screening component for the Streamlit portal."""

import io

import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors

from portal.presets import PRESETS, get_preset


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filtering criteria to predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions with columns: smiles, m, sigma, epsilon_k, in_domain, tanimoto_nn.
    filters : dict
        Filter configuration from preset.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with new column "filter_reason" (empty if passed).
    """
    df = df.copy()
    df["filter_reason"] = ""

    # m range filter
    if "m_range" in filters:
        m_min, m_max = filters["m_range"]
        mask = (df["m"] < m_min) | (df["m"] > m_max)
        df.loc[mask, "filter_reason"] += f"m out of range [{m_min}, {m_max}]; "

    # sigma range filter
    if "sigma_range" in filters:
        sig_min, sig_max = filters["sigma_range"]
        mask = (df["sigma"] < sig_min) | (df["sigma"] > sig_max)
        df.loc[mask, "filter_reason"] += f"sigma out of range [{sig_min}, {sig_max}]; "

    # epsilon_k range filter
    if "epsilon_k_range" in filters:
        eps_min, eps_max = filters["epsilon_k_range"]
        mask = (df["epsilon_k"] < eps_min) | (df["epsilon_k"] > eps_max)
        df.loc[mask, "filter_reason"] += f"epsilon_k out of range [{eps_min}, {eps_max}]; "

    # Molecular weight filter
    if "max_mw" in filters:
        max_mw = filters["max_mw"]
        df["mw"] = df["smiles"].apply(
            lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))
            if Chem.MolFromSmiles(s)
            else 0
        )
        mask = df["mw"] > max_mw
        df.loc[mask, "filter_reason"] += f"MW > {max_mw}; "

    # In-domain filter
    if filters.get("require_in_domain", False):
        mask = ~df["in_domain"]
        df.loc[mask, "filter_reason"] += "out of domain; "

    # Tanimoto filter
    if "min_tanimoto" in filters and filters["min_tanimoto"] > 0:
        min_tan = filters["min_tanimoto"]
        mask = df["tanimoto_nn"] < min_tan
        df.loc[mask, "filter_reason"] += f"Tanimoto < {min_tan}; "

    # Mark as passed if no filter reasons
    df["passed_filters"] = df["filter_reason"] == ""

    return df


def _rank_molecules(
    df: pd.DataFrame, metric: str, reference_params: dict | None = None
) -> pd.DataFrame:
    """Rank molecules by the specified metric.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered predictions.
    metric : str
        Ranking metric: "distance_to_reference", "tanimoto", "epsilon_k", "m", "sigma".
    reference_params : dict | None
        Reference molecule parameters (for distance_to_reference only).

    Returns
    -------
    pd.DataFrame
        DataFrame with new column "rank_score" (lower is better).
    """
    df = df.copy()

    if metric == "distance_to_reference":
        if not reference_params:
            st.error("Reference parameters required for distance_to_reference metric")
            df["rank_score"] = 0.0
            return df

        # Normalize parameters to [0, 1] scale using typical ranges
        # m: 1-5, sigma: 2-6, epsilon_k: 100-500
        norm_pred_m = (df["m"] - 1.0) / 4.0
        norm_pred_sigma = (df["sigma"] - 2.0) / 4.0
        norm_pred_eps = (df["epsilon_k"] - 100.0) / 400.0

        norm_ref_m = (reference_params["m"] - 1.0) / 4.0
        norm_ref_sigma = (reference_params["sigma"] - 2.0) / 4.0
        norm_ref_eps = (reference_params["epsilon_k"] - 100.0) / 400.0

        # Euclidean distance in normalized space
        df["rank_score"] = np.sqrt(
            (norm_pred_m - norm_ref_m) ** 2
            + (norm_pred_sigma - norm_ref_sigma) ** 2
            + (norm_pred_eps - norm_ref_eps) ** 2
        )

    elif metric == "tanimoto":
        # Higher Tanimoto is better → invert for ranking (lower score = better)
        df["rank_score"] = 1.0 - df["tanimoto_nn"]

    elif metric == "epsilon_k":
        # Higher epsilon_k is better (for solvents) → invert
        df["rank_score"] = -df["epsilon_k"]

    elif metric == "m":
        # Higher m is better → invert
        df["rank_score"] = -df["m"]

    elif metric == "sigma":
        # Higher sigma is better → invert
        df["rank_score"] = -df["sigma"]

    else:
        st.error(f"Unknown ranking metric: {metric}")
        df["rank_score"] = 0.0

    return df


def render_screener(api_client):
    """Render the batch screening page.

    Parameters
    ----------
    api_client : PCSAFTClient
        API client for making predictions.
    """
    st.header("Batch Molecular Screening")

    st.markdown(
        """
        Upload a CSV with candidate molecules and screen them against
        application-specific criteria. Select a preset or customize filters.
        """
    )

    # Preset selection
    st.subheader("Select Application Preset")

    preset_names = list(PRESETS.keys())
    preset_display = [PRESETS[p]["name"] for p in preset_names]

    selected_preset_idx = st.selectbox(
        "Application domain:",
        range(len(preset_names)),
        format_func=lambda i: preset_display[i],
        index=0,  # Default to blowing_agents
    )
    selected_preset_key = preset_names[selected_preset_idx]
    preset = get_preset(selected_preset_key)

    st.info(f"**{preset['name']}**: {preset['description']}")
    st.caption(f"⚠️ {preset['caution']}")

    # Show filters
    with st.expander("View/Customize Filters", expanded=False):
        filters = preset["filters"].copy()

        st.markdown("**Parameter Ranges**")
        col1, col2 = st.columns(2)

        if "m_range" in filters:
            m_min, m_max = filters["m_range"]
            m_min_new = col1.number_input("m min", value=m_min, step=0.1)
            m_max_new = col2.number_input("m max", value=m_max, step=0.1)
            filters["m_range"] = (m_min_new, m_max_new)

        if "sigma_range" in filters:
            sig_min, sig_max = filters["sigma_range"]
            sig_min_new = col1.number_input("sigma min (Å)", value=sig_min, step=0.1)
            sig_max_new = col2.number_input("sigma max (Å)", value=sig_max, step=0.1)
            filters["sigma_range"] = (sig_min_new, sig_max_new)

        if "epsilon_k_range" in filters:
            eps_min, eps_max = filters["epsilon_k_range"]
            eps_min_new = col1.number_input("epsilon_k min (K)", value=eps_min, step=10.0)
            eps_max_new = col2.number_input("epsilon_k max (K)", value=eps_max, step=10.0)
            filters["epsilon_k_range"] = (eps_min_new, eps_max_new)

        st.markdown("**Other Filters**")

        if "max_mw" in filters:
            filters["max_mw"] = st.number_input(
                "Max molecular weight",
                value=filters["max_mw"],
                step=10,
            )

        if "require_in_domain" in filters:
            filters["require_in_domain"] = st.checkbox(
                "Require in-domain (Tanimoto-based AD)",
                value=filters["require_in_domain"],
            )

        if "min_tanimoto" in filters:
            filters["min_tanimoto"] = st.slider(
                "Minimum Tanimoto similarity to training set",
                min_value=0.0,
                max_value=1.0,
                value=filters["min_tanimoto"],
                step=0.05,
            )

    st.divider()

    # Reference molecule selection
    st.subheader("Select Reference Molecule")
    ref_molecules = api_client.get_reference_molecules()

    if not ref_molecules:
        st.warning("Could not load reference molecules. Using default (Cyclopentane).")
        ref_m, ref_sigma, ref_eps = 2.3655, 3.7114, 288.84
    else:
        ref_names = [m["name"] for m in ref_molecules]
        default_ref = preset.get("default_reference", "Cyclopentane")
        default_idx = ref_names.index(default_ref) if default_ref in ref_names else 0

        selected_ref = st.selectbox(
            "Compare against:",
            ref_names,
            index=default_idx,
        )

        mol = next(m for m in ref_molecules if m["name"] == selected_ref)
        ref_m, ref_sigma, ref_eps = mol["m"], mol["sigma"], mol["epsilon_k"]
        st.caption(
            f"SMILES: `{mol['smiles']}` | m={ref_m:.4f}, "
            f"σ={ref_sigma:.4f} Å, ε/k={ref_eps:.2f} K"
        )

    reference_params = {"m": ref_m, "sigma": ref_sigma, "epsilon_k": ref_eps}

    st.divider()

    # File upload
    st.subheader("Upload Candidate Molecules")

    uploaded_file = st.file_uploader(
        "Upload CSV with a 'smiles' column",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin screening.")
        return

    try:
        df_input = pd.read_csv(uploaded_file)
        if "smiles" not in df_input.columns:
            st.error("CSV must have a 'smiles' column")
            return

        smiles_list = df_input["smiles"].dropna().astype(str).tolist()
        st.success(f"Loaded {len(smiles_list)} molecules from CSV")

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    if st.button("Run Screening", type="primary"):
        with st.spinner("Predicting PC-SAFT parameters..."):
            result = api_client.predict(smiles_list)

        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
            st.info(
                "Make sure the FastAPI server is running "
                "or set PCSAFT_API_URL='' for local mode."
            )
            return

        predictions = result.get("predictions", [])
        if not predictions:
            st.warning("No predictions returned")
            return

        # Convert to DataFrame
        df_preds = pd.DataFrame(predictions)

        # Filter valid predictions only
        df_valid = df_preds[df_preds["valid"] == True].copy()  # noqa: E712
        if len(df_valid) == 0:
            st.error("No valid predictions (all SMILES invalid)")
            return

        st.success(f"Predicted {len(df_valid)} valid molecules")

        # Apply filters
        df_filtered = _apply_filters(df_valid, filters)

        # Rank molecules
        df_ranked = _rank_molecules(
            df_filtered,
            metric=preset["ranking_metric"],
            reference_params=(
                reference_params
                if preset["ranking_metric"] == "distance_to_reference"
                else None
            ),
        )

        # Sort by rank score
        df_ranked = df_ranked.sort_values("rank_score")

        # Display screening funnel
        st.subheader("Screening Funnel")

        col1, col2, col3 = st.columns(3)
        col1.metric("Input molecules", len(smiles_list))
        col2.metric("Valid predictions", len(df_valid))
        col3.metric("Passed filters", df_ranked["passed_filters"].sum())

        # Display results
        st.divider()
        st.subheader("Screening Results")

        # Top 10 candidates
        df_top = df_ranked[df_ranked["passed_filters"]].head(10)

        if len(df_top) == 0:
            st.warning("No molecules passed filters. Try relaxing filter criteria.")
        else:
            ranking = preset['ranking_metric']
            st.markdown(f"**Top {len(df_top)} candidates** (ranked by {ranking}):")

            # Display table
            display_cols = [
                "smiles", "m", "sigma", "epsilon_k",
                "tanimoto_nn", "in_domain", "rank_score",
            ]
            st.dataframe(
                df_top[display_cols],
                use_container_width=True,
                hide_index=True,
            )

        # Show filter summary
        with st.expander("View Filter Breakdown"):
            df_failed = df_ranked[~df_ranked["passed_filters"]]
            if len(df_failed) > 0:
                st.markdown(f"**{len(df_failed)} molecules failed filters:**")
                st.dataframe(
                    df_failed[["smiles", "filter_reason"]].head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("All molecules passed filters")

        # Download enriched CSV
        st.divider()
        st.subheader("Download Results")

        csv_buffer = io.StringIO()
        df_ranked.to_csv(csv_buffer, index=False)

        st.download_button(
            label="Download enriched CSV with predictions and filters",
            data=csv_buffer.getvalue(),
            file_name=f"screening_results_{selected_preset_key}.csv",
            mime="text/csv",
        )

        st.caption(
            "The enriched CSV includes all predictions, filter results, "
            "and ranking scores. Use this for further analysis or external validation."
        )
