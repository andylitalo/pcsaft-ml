"""Model dashboard component for the Streamlit portal."""

import streamlit as st


def render_dashboard(api_client):
    """Render the model performance dashboard.

    Parameters
    ----------
    api_client : PCSAFTClient
        API client for fetching model info.
    """
    st.header("Model Dashboard")

    st.markdown(
        """
        Monitor the current model's performance and training data status.
        """
    )

    # Fetch model info
    with st.spinner("Loading model info..."):
        info = api_client.get_model_info()

    # System health
    status = info.get("status", "unknown")
    if status == "healthy":
        st.success("✅ API is healthy")
    elif status == "offline":
        st.warning("⚠️ API is offline (using cached metrics)")
    else:
        st.error(f"❌ System status: {status}")

    # Current model
    st.subheader("Current Model")
    col1, col2 = st.columns(2)
    col1.metric("Model Type", info.get("model_name", "unknown"))
    col2.metric("Model Loaded", "Yes" if info.get("model_loaded", False) else "No")

    # Performance metrics
    st.subheader("Model Performance (Test Set)")

    st.markdown(
        """
        **R² scores** measure prediction quality (1.0 = perfect, 0.0 = no better than mean).
        Higher is better.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("R² (m)", f"{info.get('r2_m', 0):.3f}")
    col2.metric("R² (σ)", f"{info.get('r2_sigma', 0):.3f}")
    col3.metric("R² (ε/k)", f"{info.get('r2_epsilon_k', 0):.3f}")

    st.markdown("**Mean Absolute Error (MAE)** in natural units. Lower is better.")

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (m)", f"{info.get('mae_m', 0):.3f}")
    col2.metric("MAE (σ)", f"{info.get('mae_sigma', 0):.3f} Å")
    col3.metric("MAE (ε/k)", f"{info.get('mae_epsilon_k', 0):.1f} K")

    # Training data
    st.subheader("Training Data")
    st.metric("Total training molecules", info.get("n_training", 0))

    # Notes on performance
    with st.expander("About these metrics"):
        st.markdown(
            """
            ### Model Performance Notes

            - **m (segment number)**: Easiest parameter to predict (R² ~ 0.6–0.7)
            - **σ (segment diameter)**: Moderate difficulty (R² ~ 0.3–0.4)
            - **ε/k (dispersion energy)**: Hardest to predict (R² ~ 0.3–0.4)

            Baseline metrics: Random Forest on RDKit 2D descriptors + Morgan
            fingerprints. Steps 2–4 explore neural networks and transformers.

            ### How to Improve Performance

            1. **Submit experimental data** via the "Submit Experimental Data" tab
            2. System automatically retrains when enough new data accumulates
            3. Better models (NN, ChemBERTa) are available via `MODEL_TYPE` env var
            """
        )
