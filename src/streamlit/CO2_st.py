# =====================================================================================
# LIBRARIES SECTION
# =====================================================================================

import streamlit as st
import gdown  # to load data from Tillmann's google drive
import time
import joblib  # For saving and loading models
import os

# Import standard libraries
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt
#import seaborn as sns

# Import libraries for the modeling
#from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.linear_model import ElasticNetCV
#import xgboost as xgb

# TensorFlow for DNN
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Input
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import EarlyStopping


# =====================================================================================
# STREAMLIT OVERALL STRUCTURE
# =====================================================================================

st.title("CO2 emissions by vehicles")
pages = ["Home", "Data and Preprocessing", "Choice of Models", "Evaluating Model Performance", "Conclusions"]
page = st.sidebar.radio("Go to", pages)

# =====================================================================================
# HOME SECTION
# =====================================================================================

if page == pages[0]:
    st.write('## Context')

    image_url = "https://www.mcilvainmotors.com/wp-content/uploads/2024/02/Luxury-Car-Increased-Emission.jpg"
    st.image(image_url, caption="Luxury Car Emissions", use_container_width=True)
    
    # Display the text
    st.write("Identifying the vehicles that emit the most CO2 is important to identify the technical characteristics that play a role in pollution.") 
    st.write("Predicting this pollution in advance makes it possible to prevent the appearance of new types of vehicles (new series of cars for example.)")

    st.write('**Project Purpose and Goals**')
    st.write("This project focuses on using machine learning to help the automotive industry meet the EU's 2035 target of 0 g CO₂/km for passenger cars and vans. By analyzing extensive vehicle data, machine learning models can identify factors influencing CO₂ emissions, aiding manufacturers in designing low-emission vehicles. This ensures compliance with regulations, reduces penalties, and enhances brand reputation. The project also aims to optimize production strategies by enabling early design adjustments, especially as the industry shifts towards zero-emission vehicles and considers alternative energy sources like hydrogen or electricity for appliances in motorhomes (as a collateral effect)")

# =====================================================================================
# DATA PREPROCESSING SECTION
# =====================================================================================

# Google drive file link of final preprocessed data set
file_url = "https://drive.google.com/uc?id=13hNrvvMgxoxhaA9xM4gmnSQc1U2Ia4i0"  # file link to Tillmann's drive
output = 'data.parquet'

# Check if data is already loaded in session state, if not, load it
if "df" not in st.session_state:
    st.session_state.df = None  # Initialize df in session state

if page == pages[1]:  # Only load data when on the "Data and Preprocessing" page
    try:
        # Try downloading the file from Google Drive
        gdown.download(file_url, output, quiet=False)

        # Load the data into a DataFrame
        st.session_state.df = pd.read_parquet(output)
        st.write("Data loaded successfully from Google Drive")

    except Exception as e:
        # If Google Drive download fails, load data from local path
        st.write("Failed to load data from Google Drive. Reverting to local data.")
        st.session_state.df = pd.read_parquet(r'C:\Users\alexa\Downloads\ProjectCO2--no Github\Data\minimal_withoutfc_dupesdropped_frequencies_area_removedskew_outliers3_0_NoW_tn20_mcp00.10.parquet')

    # Display the data
    st.write('Presentation of Data and Preprocessing')
    st.write("Data Loaded Successfully", st.session_state.df.head())

# =====================================================================================
# CHOICE OF MODELS SECTION
# =====================================================================================

if page == pages[2]:
    st.write("## Selecting the appropriate algorithm")
    image_url = "https://blog.medicalalgorithms.com/wp-content/uploads/2015/03/medical-algorithm-definition.jpg"
    st.image(image_url, caption="Machine learning, deep learning?", use_container_width=True)

    st.write("Since our project involves predicting a continuous target variable, Worldwide Harmonised Light vehicles Test Procedure (Ewltp (g/km)), this is inherently a regression problem.") 
    st.write("Our primary approach was to establish a robust baseline using multilinear regression, a widely accepted model for regression tasks.") 
    st.write("This choice allowed us to evaluate the model's performance under straightforward, interpretable assumptions about linear relationships between features and the target variable.") 

    st.write('**First models**')
    st.write("1- **Linear Regression with Elastic Net**")
    st.write("2- **Decision Trees**: chosen for their interpretability and ease of handling non-linear relationships. However, prone to overfitting.")
    st.write("3- **Random Forests**: as an ensemble method that aggregates multiple Decision Trees. They are more robust and show reduced tendency to overfit compared to a single Decision Tree.")
    st.write("4- **XGBoost**: also a tree-based ensemble method, which improves performance by sequentially building trees and learning from residual errors")
    st.write("5- **Dense Neural Networks**: lastly introduced as a deep learning approach to explore the possibility of capturing highly complex interactions among features that may not be adequately handled by tree-based algorithms")

    st.write('**Final models: Exclusion of Decision Trees and Random Forests**')
    st.markdown("""- Redundancy with XGBoost: it's advanced algorithm that surpasses the performance of Decision Trees and Random Forests.
                - In our tests, XGBoost not only yielded higher accuracy but also demonstrated more stable performance across various data splits.
                - XGBoost is optimized for scalability with large datasets and offers greater control over hyperparameters, making it better suited for fine-tuning (Literature). 
""")

    st.write('**Optimization and Evaluation Techniques**')
    st.write("The table below provides an overview of the optimization and evaluation used for each model, along with interpretability methods")

    # Define the table in Markdown format
    markdown_table = """
    | Models/Techniques       | Grid Search              | Elastic Net                               | Cross Validation                  | Interpretability    |
    |-------------------------|--------------------------|-------------------------------------------|-----------------------------------|---------------------|
    | Linear Regression       | No                       | Yes (given persistent multicollinearity)  | Yes (to evaluate generalizability) | Feature Importance  |
    | XG Boost                | Yes (opt. parameters)    | Not applicable                            | Yes (evaluate generalizability)   | Shap values         |
    | Dense Neural Network    | No                       | Not applicable, but Ridge regularization was applied | No, but a validation set was used. | Not applied         |
    """

    # Display the table in Streamlit
    st.markdown(markdown_table)



    st.write('**Interpretability**')

    # Display images showcasing feature importance for Linear Regression and XGBoost
    #linear_regression_image_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\visualization\Feature Importance Linear Regression.png'
    #xgboost_image_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\visualization\Feature Importance XG Boost.png'

    # Show images in Streamlit
    #st.image(linear_regression_image_path, caption="Feature Importance - Linear Regression", use_column_width=True)
    #st.image(xgboost_image_path, caption="SHAP Values - XGBoost", use_column_width=True)

    # Google Drive URLs for the images
    linear_regression_image_url = "https://drive.google.com/uc?id=1JWR6BqH8eebiZmtyLOgslDDGHGOq4ec3"  # Feature Importance for Linear Regression
    xgboost_image_url = "https://drive.google.com/uc?id=14iFU17b6_wMzsYNTdtda9ZsOrbp0Uq7D"  # Feature Importance for XGBoost

    # Download the images
    linear_regression_image_path = "linear_regression_feature_importance.png"
    xgboost_image_path = "xgboost_feature_importance.png"

    gdown.download(linear_regression_image_url, linear_regression_image_path, quiet=False)
    gdown.download(xgboost_image_url, xgboost_image_path, quiet=False)

    # Show images in Streamlit
    st.image(linear_regression_image_path, caption="Feature Importance - Linear Regression", use_container_width=True)
    st.image(xgboost_image_path, caption="SHAP Values - XGBoost", use_container_width=True)


# =====================================================================================
# MODELISATION AND ANALYSIS SECTION
# =====================================================================================
if page == pages[3]:
    st.write("Modelisation")

    # Define the metrics for each model
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

    # Show the checkboxes for each model's metrics
    st.write("### Choose models to display metrics:")

    model_checkboxes = {
        "Linear Regression": st.checkbox("Linear Regression"),
        "XG Boost": st.checkbox("XG Boost"),
        "Dense Neural Network": st.checkbox("Dense Neural Network")
    }

    # Display the selected model metrics
    selected_models = []
    for model, checkbox in model_checkboxes.items():
        if checkbox:
            selected_models.append(model)
            st.write(f"**{model}**")
            for metric, value in model_results[model].items():
                st.write(f"{metric}: {value}")

    # Show comparison table when checkbox is selected
    if st.checkbox('Show all models comparison'):
        if selected_models:
            comparison_data = {model: model_results[model] for model in selected_models}
            comparison_df = pd.DataFrame(comparison_data).T
            st.write(comparison_df)
        else:
            st.write("Please select at least one model to compare.")


# =====================================================================================
# Conclusion
# =====================================================================================


if page == pages[4]:
    st.write("## Conclusion about the models")

    st.write("""
    **XGBoost** emerges as the strongest model in terms of predictive accuracy and robustness, with the lowest Mean Squared Error (MSE), highest R-squared, and a reasonable training time. Its decision tree-based structure allows it to effectively capture non-linear relationships, which linear regression cannot. XGBoost’s balance between performance and efficiency makes it a highly suitable choice for this task.

    **Dense Neural Network (DNN)** is a close runner-up, offering fast training times and high R-squared. However, as noted previously, it may require more fine-tuning to prevent overfitting or instability across different data subsets. Despite this, its deep learning approach offers potential for further improvement with better regularization.

    Although **Linear Regression** provides simplicity and interpretability, its lower performance metrics suggest that it may not be sufficient for this dataset. While it serves as a useful benchmark, it falls short compared to the flexibility and accuracy of the more advanced models like XGBoost and DNN.

    In conclusion, **XGBoost** is recommended as the primary model due to its optimal balance of accuracy, stability, and efficiency. It outperforms other models while offering the best compromise between computational efficiency and predictive power, making it the most suitable for deployment in this context.
    """)


    st.write("## Conclusion about subject matter")
    st.write("""Unsurprisingly, reducing autofeatures such as Weight, size, engine capacity (ccm), engine power, as well as, adopting single or better combined innovative technologies like exterior LED lights and/or 12 volt efficient alternators will reduce the CO2 emissions.
                In general terms, shifting to electric powered vehicles emitting by definition 0 gr/km CO2 is the way to go and which has been adopted by almost all car manufacturers.""")


    st.write("## Prospects for improvement") 
    st.write("""
    **Ensemble Methods**: Combining predictions from models like Linear Regression, XGBoost, and Dense Neural Networks (DNN) in a stacked ensemble could leverage their strengths and enhance overall accuracy.

    **Hyperparameter Tuning**: While grid search optimized some models, more exhaustive tuning or Bayesian Optimization could improve performance, particularly for complex models like XGBoost and DNN.

    **Deep Learning Architectures**: Exploring architectures like Convolutional Neural Networks (CNNs) for structured data or Residual Connections/Layer Normalization could reduce overfitting and capture more nuanced relationships.

    **Cross-validation with Time Series Splits**: Time-based cross-validation could improve generalization across different time periods, ensuring better model performance on future data.

    **Data Augmentation**: If certain categories are underrepresented, techniques like SMOTE could balance the dataset by creating synthetic data, improving model robustness.
    """)

