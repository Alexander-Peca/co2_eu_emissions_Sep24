# =====================================================================================
# LIBRARIES SECTION
# =====================================================================================

import streamlit as st
import gdown  # to load data from Tillmann's google drive
import os
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join('..')))

# Import self-defined functions
from utils_CO2 import *

# Import standard libraries
import pandas as pd
import numpy as np


# =====================================================================================
# STREAMLIT OVERALL STRUCTURE
# =====================================================================================

st.title("CO2 Emissions by Vehicles")
pages = ["Home", "Data and Preprocessing", "Choice of Models", "Evaluating Model Performance", "Conclusions"]
page = st.sidebar.radio("Navigate to:", pages)

# =====================================================================================
# HOME SECTION
# =====================================================================================

if page == pages[0]:
    st.write('## Context')

    # Adding an image to give context
    image_url = "https://www.mcilvainmotors.com/wp-content/uploads/2024/02/Luxury-Car-Increased-Emission.jpg"
    st.image(image_url, caption="Luxury Car Emissions", use_container_width=True)

    # Description text
    st.write("""
        **Understanding the problem:**
        Identifying the vehicles that emit the most CO2 is crucial to understanding the technical characteristics that contribute to pollution. 
        Predicting this pollution in advance helps prevent the development of new types of vehicles that may worsen emissions.
    """)

    # Project Purpose and Goals
    st.write('### Project Purpose and Goals')
    st.write("""
        This project uses machine learning to help the automotive industry meet the EU’s 2035 target of 0 g CO₂/km for passenger cars and vans. 
        By analyzing extensive vehicle data, we aim to identify factors influencing CO₂ emissions, helping manufacturers design low-emission vehicles. 
        Additionally, the project optimizes production strategies, enabling early design adjustments as the industry shifts toward zero-emission vehicles.
    """)

# =====================================================================================
# DATA PREPROCESSING SECTION
# =====================================================================================

file_url = "https://drive.google.com/uc?id=13hNrvvMgxoxhaA9xM4gmnSQc1U2Ia4i0"  # file link to Tillmann's drive
output = 'data.parquet'

# Google Drive URLs for the images
# linear_regression_image_url = "https://drive.google.com/uc?id=1JWR6BqH8eebiZmtyLOgslDDGHGOq4ec3"  # Feature Importance for Linear Regression
linear_regression_image_url = "https://drive.google.com/file/d/1nO5_SZ8EBZ7qcrcKo2uJ_U_hJeDnPoUP"
xgboost_image_url = "https://drive.google.com/uc?id=14iFU17b6_wMzsYNTdtda9ZsOrbp0Uq7D"  # Feature Importance for XGBoost



# Check if data is already loaded in session state, if not, load it
# if "df" not in st.session_state:
#    st.session_state.df = None  # Initialize df in session state

if page == pages[1]:
    if "df" not in st.session_state:
        try:
            gdown.download(file_url, output, quiet=False)
            st.session_state.df = pd.read_parquet(output)
            st.write("Data loaded successfully from Google Drive")

        except Exception as e:
            st.write("Failed to load data from Google Drive. Reverting to local data.")
            st.session_state.df = pd.read_parquet(r'C:\Users\alexa\Downloads\ProjectCO2--no Github\Data\minimal_withoutfc_dupesdropped_frequencies_area_removedskew_outliers3_0_NoW_tn20_mcp00.10.parquet')

    df = st.session_state.df
    st.write('### Presentation of Data and Preprocessing')
    st.write("Data Loaded Successfully", df.head(),df['Mh_Other'].value_counts())

    
    # get sorted column names
    columns = sort_columns(df)
    
    ######################################################
    # Create an HTML string with column names separated by commas
    columns_html = ", ".join([f"<span>{col}</span>" for col in columns])
    # Display using markdown with unsafe_allow_html=True
    st.markdown(f"**Columns in DataSet:** {columns_html}", unsafe_allow_html=True)
    
    ########################################
    with st.expander("Show Columns in DataSet"):
        columns_str = ", ".join(columns)
        st.write(columns_str)
   


# =====================================================================================
# CHOICE OF MODELS SECTION
# =====================================================================================

if page == pages[2]:
    st.write("## Selecting the Appropriate Algorithm")

    image_url = "https://blog.medicalalgorithms.com/wp-content/uploads/2015/03/medical-algorithm-definition.jpg"
    st.image(image_url, caption="Machine Learning and Deep Learning?", use_container_width=True)

    st.write("""
        Since the task involves predicting a continuous target variable, the Worldwide Harmonised Light vehicles Test Procedure (Ewltp (g/km)),
        this is inherently a regression problem. Our primary approach was to establish a robust baseline using multilinear regression, 
        a widely accepted model for regression tasks.
    """)

    st.write('### First Models')
    st.write("""
        1. **Linear Regression with Elastic Net**  
        2. **Decision Trees**: Easy to interpret but prone to overfitting.  
        3. **Random Forests**: Ensemble method reducing overfitting.
        4. **XGBoost**: Sequential tree-based method that captures residual errors for better performance.
        5. **Dense Neural Networks**: Deep learning approach exploring complex relationships.
    """)

    st.write('### Final Models: Exclusion of Decision Trees and Random Forests')
    st.markdown("""
        - **Redundancy with XGBoost**: XGBoost surpasses Decision Trees and Random Forests in accuracy and stability.
        - **XGBoost Optimization**: Offers greater scalability with large datasets and better hyperparameter control.
    """)

    st.write('### Optimization and Evaluation Techniques')
    st.write("""
        Below is an overview of the optimization techniques used for each model, along with interpretability methods.
    """)

    markdown_table = """
    | Models/Techniques       | Grid Search              | Elastic Net                               | Cross Validation                  | Interpretability    |
    |-------------------------|--------------------------|-------------------------------------------|-----------------------------------|---------------------|
    | Linear Regression       | No                       | Yes (persistent multicollinearity)  | Yes (generalizability) | Feature Importance  |
    | XG Boost                | Yes (opt. parameters)    | Not applicable                            | Yes (generalizability)   | Shap values         |
    | Dense Neural Network    | No                       | Not applicable, Ridge regularization | No (validation set used) | Not applied         |
    """

    st.markdown(markdown_table)


    st.write('### Interpretability')
    
    # Google Drive URLs for the images
    # linear_regression_image_url = "https://drive.google.com/uc?id=1JWR6BqH8eebiZmtyLOgslDDGHGOq4ec3"  # Feature Importance for Linear Regression
    linear_regression_image_url = "https://drive.google.com/uc?id=1nO5_SZ8EBZ7qcrcKo2uJ_U_hJeDnPoUP"  # # Feature Importance for Linear Regression_signed_weighted
    xgboost_image_url = "https://drive.google.com/uc?id=14iFU17b6_wMzsYNTdtda9ZsOrbp0Uq7D"  # Feature Importance for XGBoost
    dnn_image_url = "https://drive.google.com/uc?id=1TzyMGuRzpJnLEidZpsZbcnHVO42tMEYh"      # Feature Importance for DNN

    # Download the images
    linear_regression_image_path = "linear_regression_feature_importance.png"
    xgboost_image_path = "xgboost_feature_importance.png"
    dnn_image_path = "dnn_feature_importance.png"

    gdown.download(linear_regression_image_url, linear_regression_image_path, quiet=False)
    gdown.download(xgboost_image_url, xgboost_image_path, quiet=False)
    gdown.download(dnn_image_url, dnn_image_path, quiet=False)

    # Show images in Streamlit
    st.image(linear_regression_image_path, caption="Feature Importance - Linear Regression", use_container_width=True)
    st.image(xgboost_image_path, caption="SHAP Values - XGBoost", use_container_width=True)
    st.image(dnn_image_path, caption="Weights First Layer - DNN", use_container_width=True)


# =====================================================================================
# MODELISATION AND ANALYSIS SECTION
# =====================================================================================

if page == pages[3]:
    st.write("## Modelisation")

    # Metrics for each model
    model_results = {
        "Linear Regression": {
            "Test MSE": 0.126959,
            "Test R-squared": 0.873043,
            "CV R-squared": 0.873155,
            "Training time (mins)": 7.07
        },
        "XG Boost": {
            "Test MSE": 0.026375,
            "Test R-squared": 0.973626,
            "CV R-squared": 0.973652,
            "Training time (mins)": 1.41
        },
        "Dense Neural Network": {
            "Test MSE": 0.061685,
            "Test R-squared": 0.938316,
            "CV R-squared": "N/A",
            "Training time (mins)": 0.26
        }
    }

    # Display comparison table when checkbox is selected
    st.write("### Show all models comparison:")
    
    show_comparison = st.checkbox('Show all models comparison')

    if show_comparison:
        comparison_df = pd.DataFrame(model_results).T
        st.write(comparison_df)

    # Metrics for individual models
    st.write("### Choose models to display individual metrics:")

    model_checkboxes = {
        "Linear Regression": st.checkbox("Linear Regression"),
        "XG Boost": st.checkbox("XG Boost"),
        "Dense Neural Network": st.checkbox("Dense Neural Network")
    }

    for model, checkbox in model_checkboxes.items():
        if checkbox:
            st.write(f"**{model}**")
            for metric, value in model_results[model].items():
                st.write(f"{metric}: {value}")

# =====================================================================================
# CONCLUSIONS SECTION
# =====================================================================================

if page == pages[4]:
    st.write("## Conclusion about the Models")

    st.write("""
    **XGBoost** emerges as the strongest model in terms of predictive accuracy, with the lowest Mean Squared Error (MSE) and the highest R-squared. 
    It effectively captures non-linear relationships that Linear Regression cannot, offering an optimal balance of performance and efficiency.
    """)

    st.write("""
    **Dense Neural Network (DNN)** is a close runner-up, with fast training times and high R-squared. However, it requires further fine-tuning to prevent overfitting.
    """)

    st.write("""
    Although **Linear Regression** provides interpretability and simplicity, its performance metrics suggest it is not enough for this dataset. 
    It is useful as a benchmark, but it falls behind compared to XGBoost and DNN.
    """)

    st.write("""
    In conclusion, **XGBoost** is recommended as the primary model due to its optimal balance of accuracy, stability, and efficiency.
    """)

    st.write("## Conclusion about the Subject Matter")
    st.write("""
    Shifting to electric-powered vehicles emitting 0 CO₂/km is the future, as car manufacturers are increasingly adopting this standard.
    """)

    st.write("## Prospects for Improvement")
    st.write("""
    - **Ensemble Methods**: Combining predictions from Linear Regression, XGBoost, and Dense Neural Networks could improve accuracy.
    - **Hyperparameter Tuning**: More exhaustive tuning like Bayesian Optimization could optimize XGBoost and DNN.
    - **Deep Learning Architectures**: Exploring CNNs or Residual Connections could improve model performance.
    - **Cross-validation with Time Series Splits**: Ensures better model generalization across different periods.
    - **Data Augmentation**: Techniques like SMOTE could balance underrepresented categories in the dataset.
    """)

