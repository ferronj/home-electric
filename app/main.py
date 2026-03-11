"""Streamlit app entry point."""

import streamlit as st

st.set_page_config(
    page_title="Electricity Usage Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.sidebar import render_sidebar
from app.pages.dashboard import render_dashboard
from app.pages.modeling import render_modeling

# Sidebar
app_state = render_sidebar()

# Main content
st.title("Electricity Usage Analysis")

tab_dashboard, tab_modeling = st.tabs(["Dashboard", "Bayesian Modeling"])

with tab_dashboard:
    render_dashboard(app_state)

with tab_modeling:
    render_modeling(app_state)
