"""Streamlit application for PC-SAFT parameter prediction portal."""

import os

import streamlit as st

from portal.api_client import PCSAFTClient
from portal.components import render_dashboard, render_prediction_page, render_submission_page

# Configure page
st.set_page_config(
    page_title="PC-SAFT Parameter Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize API client
api_url = os.getenv("PCSAFT_API_URL", "http://localhost:8000")
api_client = PCSAFTClient(base_url=api_url)

# Main title
st.title("🧪 PC-SAFT Parameter Prediction Portal")

st.markdown(
    f"""
    Predict equation-of-state parameters for potential blowing agents.
    Compare candidates to **cyclopentane** (m=2.37, σ=3.71 Å, ε/k=289 K).

    *Connected to API: `{api_url}`*
    """
)

# Create tabs
tab_predict, tab_submit, tab_dashboard = st.tabs(
    [
        "🔮 Predict Parameters",
        "📊 Submit Experimental Data",
        "📈 Model Dashboard",
    ]
)

with tab_predict:
    render_prediction_page(api_client)

with tab_submit:
    render_submission_page(api_client)

with tab_dashboard:
    render_dashboard(api_client)

# Footer
st.divider()
st.caption(
    "PC-SAFT Parameter Predictor | "
    "ML-driven screening for blowing agent candidates | "
    "Powered by Random Forest, PyTorch, and ChemBERTa"
)
