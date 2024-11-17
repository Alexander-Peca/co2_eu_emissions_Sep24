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
    st.write("Data Loaded Successfully")

    
    # get sorted column names
    columns = sort_columns(df)
    
    ####################################################
    with st.expander("Show all Columns in DataSet"):
        columns_str = ", ".join(columns)
        st.write(columns_str)
        
        
    ###################################################
    # List of tuples with prefixes and baseline categories
    categorical_info = [
    ('Ct', 'Ct_M1 (Baseline)'),
    ('Fm', 'Fm_B (Baseline)'),
    ('Ft', 'Ft_DIESEL (Baseline)'),
    ('Mh', 'Mh_AA-IVA (Baseline)'),
    ('IT_code', 'IT_code_None (Baseline)')
    ]

    numerical_summary, categorical_summaries = create_summary_tables(df, categorical_info)
        
    st.header("Numerical Summary")
    st.dataframe(numerical_summary)

    st.header("Categorical Summaries")
    for prefix, summary_table in categorical_summaries.items():
        st.markdown(f"**Attribute {prefix}**")  # Regular text size with bold formatting
        st.dataframe(summary_table)

   


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
import requests
from io import BytesIO

if page == pages[3]:
    st.write("## Modelisation")

    # Metrics for each model
    model_results = {
        "Linear Regression": {
            "Test MSE (scl.)": 0.12,
            "Test MSE": 120.5,
            "Test RMSE": 11.0,
            "Test MAE": 8.3,
            "Test R²": 0.87,
            "CV R²": 0.87,
            "Training time (mins)": 7.07
        },
        "Dense Neural Network": {
            "Test MSE (scl.)": 0.06,
            "Test MSE": 64.0,
            "Test MAE": 6.2,
            "Test RMSE": 8.0,
            "Test R²": 0.93,
            "CV R²": None,
            "Training time (mins)": 0.26
        },
        "XG Boost": {
            "Test MSE (scl.)": 0.02,
            "Test MSE": 25.1,
            "Test MAE": 3.7,
            "Test RMSE": 5.0,
            "Test R²": 0.97,
            "CV R²": 0.97,
            "Training time (mins)": 1.41
        }
    }

    # Image URLs for each model
    image_urls = {
        "Linear Regression": [
            "https://drive.google.com/uc?id=13r71MniN62xtjOq4jMz8Y6rlwx7qvAv8", # LR Actual vs Predicted
            "https://drive.google.com/uc?id=1UKd5oawhUEuaURpA3iWMIt6NmSNnlikX", # LR Residuals
            "https://drive.google.com/uc?id=1evpCV76mWn8emda25Qzb7HJXHcJTgkNM", # LR QQ-Plot
            "https://drive.google.com/uc?id=1hKXCqHT9sXjk_EywRDsHFP19_-FJEBao", # LR Residuals Histogram
        ],
        "XG Boost": [
            "https://drive.google.com/uc?id=1RsdH50v7UzH3jVbh4HN7AfzLC6xeZORw", # XGB Actual vs Predicted
            "https://drive.google.com/uc?id=19A62Gz-3LGGBjaOHVVxokDxl2wxR8i_D", # XGB Residuals
            "https://drive.google.com/uc?id=1BRf6BKr_GhvlCuZYqpGYhVQs2-kJt1Ih", # XGB QQ-Plot
            "https://drive.google.com/uc?id=19uhWB2_iTA7jv83wqvTynyzPDoo0UPX1", # XGB Residuals Histogram
        ],
        "Dense Neural Network": [
            "https://drive.google.com/uc?id=1O28Nb38iz9rs0or58rwrDMgDCKTqx4Xj", # DNN Actual vs Predicted
            "https://drive.google.com/uc?id=1LIFPSTMbag94UYT7R-kbWTpwOfGyU9i5", # DNN Residuals
            "https://drive.google.com/uc?id=1ccY-04Ll8b9097fGJMX7WfnD5J6xTFG3", # DNN QQ-Plot
            "https://drive.google.com/uc?id=1rm27rWIbHKgQgVh2Adm6NEnNG9jA7qCH", # DNN Residuals Histogram
        ]
    }

    # Function to fetch and cache images
    @st.cache_data
    def fetch_image_from_url(url):
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)

    # Display comparison table when checkbox is selected
    st.write("### Show models comparison:")
    
    show_comparison = st.checkbox('Show models comparison')

    if show_comparison:
        comparison_df = pd.DataFrame(model_results).T
        st.write(comparison_df)

    # Visualizations for models
    st.write("### Choose visualizations to display models comparison:")

    show_act_vs_pred = st.checkbox("Actual vs Predicted Values")
    show_residuals = st.checkbox("Residuals: Error Distribution")
    show_qq_plot = st.checkbox("Residuals: QQ-Plot")
    show_residuals_hist = st.checkbox("Residuals: Histogram")

    # Visualization indices
    visualization_indices = {
        "Actual vs Predicted Values": 0,
        "Residuals: Error Distribution": 1,
        "Residuals: QQ-Plot": 2,
        "Residuals: Histogram": 3
    }

    # Specify the desired column order
    column_order = ["Linear Regression", "Dense Neural Network", "XG Boost"]

    # Display selected visualizations
    if any([show_act_vs_pred, show_residuals, show_qq_plot, show_residuals_hist]):
        # Loop through each visualization type
        for viz_name, viz_index in visualization_indices.items():
            if ((viz_name == "Actual vs Predicted Values" and show_act_vs_pred) or
                (viz_name == "Residuals: Error Distribution" and show_residuals) or
                (viz_name == "Residuals: QQ-Plot" and show_qq_plot) or
                (viz_name == "Residuals: Histogram" and show_residuals_hist)):

                st.write(f"### {viz_name}")

                cols = st.columns(3)  # Create 3 columns
                for i, model in enumerate(column_order):  # Use the specified column order
                    with cols[i]:
                        st.write(f"**{model}**")
                        image_data = fetch_image_from_url(image_urls[model][viz_index])
                        st.image(image_data, caption=f"{model} {viz_name}", use_container_width=True)










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

