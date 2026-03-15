"""Streamlit application for PC-SAFT parameter prediction portal."""

import os

import streamlit as st

from portal.api_client import PCSAFTClient
from portal.components import render_dashboard, render_prediction_page, render_submission_page
from portal.components.screener import render_screener

# Configure page
st.set_page_config(
    page_title="PC-SAFT Parameter Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize API client
api_url = os.getenv("PCSAFT_API_URL", "http://localhost:8000")
# Support local mode: if PCSAFT_API_URL is empty string, use None for local fallback
if api_url == "":
    api_url = None
api_client = PCSAFTClient(base_url=api_url)

# Main title
st.title("🧪 PC-SAFT Parameter Prediction Portal")

if api_url:
    st.markdown(
        f"""
        Predict equation-of-state parameters for multiple application domains:
        blowing agents, refrigerants, solvents, or general exploration.

        *Connected to API: `{api_url}`*
        """
    )
else:
    st.markdown(
        """
        Predict equation-of-state parameters for multiple application domains:
        blowing agents, refrigerants, solvents, or general exploration.

        *Running in local mode (using pcsaft_predict library directly)*
        """
    )

# Create tabs
tab_predict, tab_screen, tab_submit, tab_dashboard = st.tabs(
    [
        "🔮 Predict Parameters",
        "🎯 Screen Candidates",
        "📊 Submit Experimental Data",
        "📈 Model Dashboard",
    ]
)

with tab_predict:
    render_prediction_page(api_client)

with tab_screen:
    render_screener(api_client)

with tab_submit:
    render_submission_page(api_client)

with tab_dashboard:
    render_dashboard(api_client)

# Footer
st.divider()
st.caption(
    "PC-SAFT Parameter Predictor | "
    "ML-driven screening for multiple application domains | "
    "Powered by Random Forest, PyTorch, and ChemBERTa"
)
