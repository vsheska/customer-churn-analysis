import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os


def load_image(image_path):
    """Load an image from path."""
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None


def main():
    st.title("Customer Churn Analysis Dashboard")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Feature Analysis", "Model Performance"]
    )

    # Overview Page
    if page == "Overview":
        st.header("Project Overview")
        st.write("""
        This dashboard presents the results of our customer churn analysis experiment.
        The analysis includes feature distributions, correlations, and model performance metrics.
        """)

        # Load and display correlation matrix
        corr_matrix = load_image("outputs/plots/correlation_matrix.png")
        if corr_matrix:
            st.subheader("Correlation Matrix")
            st.image(corr_matrix, use_container_width=True)

        # Load and display churn correlations
        churn_corr = load_image("outputs/plots/churn_correlations.png")
        if churn_corr:
            st.subheader("Churn Correlations")
            st.image(churn_corr, use_container_width=True)

    # Feature Analysis Page
    elif page == "Feature Analysis":
        st.header("Feature Analysis")

        # Create columns for histograms
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Monthly Charges Distribution")
            monthly_charges = load_image("outputs/plots/histogram_MonthlyCharges.png")
            if monthly_charges:
                st.image(monthly_charges, use_container_width=True)

            st.subheader("Contract Value Distribution")
            contract_value = load_image("outputs/plots/histogram_ContractValue.png")
            if contract_value:
                st.image(contract_value, use_container_width=True)

        with col2:
            st.subheader("Tenure Distribution")
            tenure = load_image("outputs/plots/histogram_tenure.png")
            if tenure:
                st.image(tenure, use_container_width=True)

            st.subheader("Total Charges Distribution")
            total_charges = load_image("outputs/plots/histogram_TotalCharges.png")
            if total_charges:
                st.image(total_charges, use_container_width=True)

        # Display boxplots
        st.subheader("Feature Boxplots")
        boxplots = load_image("outputs/plots/boxplots.png")
        if boxplots:
            st.image(boxplots, use_container_width=True)

    # Model Performance Page
    elif page == "Model Performance":
        st.header("Model Performance Metrics")

        # Display confusion matrix
        st.subheader("Confusion Matrix")
        conf_matrix = load_image("outputs/plots/confusion_matrix.png")
        if conf_matrix:
            st.image(conf_matrix, use_container_width=True)

        # Display ROC curve
        st.subheader("ROC Curve")
        roc_curve = load_image("outputs/plots/roc_curve.png")
        if roc_curve:
            st.image(roc_curve, use_container_width=True)

        # Display cross-validation scores
        st.subheader("Cross-validation Scores")
        cv_scores = load_image("outputs/plots/cv_scores.png")
        if cv_scores:
            st.image(cv_scores, use_container_width=True)


if __name__ == "__main__":
    main()