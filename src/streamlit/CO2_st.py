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

import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for the modeling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNetCV
import xgboost as xgb

# TensorFlow for DNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# =====================================================================================
# STREAMLIT OVERALL STRUCTURE
# =====================================================================================

st.title("CO2 emissions by vehicles")
pages = ["Home", "Data and Preprocessing", "Choice of Models", "Modelisation and Analysis", "Conclusion"]
page = st.sidebar.radio("Go to", pages)

# =====================================================================================
# HOME SECTION
# =====================================================================================

if page == pages[0]:
    st.write('Context')

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

# Initialize df as None to avoid reference errors on other pages
df = None

# Google drive file link of final preprocessed data set
file_url = "https://drive.google.com/uc?id=13hNrvvMgxoxhaA9xM4gmnSQc1U2Ia4i0"  # file link to Tillmann's drive

# Output filename where the file will be saved
output = 'data.parquet'

if page == pages[1]:  # Only show messages and load data on page 1
    try:
        # Try downloading the file from Google Drive
        gdown.download(file_url, output, quiet=False)

        # Load the data into a DataFrame
        df = pd.read_parquet(output)
        st.write("Data loaded successfully from Google Drive")

    except Exception as e:
        # If Google Drive download fails, load data from local path
        st.write("Failed to load data from Google Drive. Reverting to local data.")
        df = pd.read_parquet(r'C:\Users\alexa\Downloads\ProjectCO2--no Github\Data\minimal_withoutfc_dupesdropped_frequencies_area_removedskew_outliers3_0_NoW_tn20_mcp00.10.parquet')

    # Display the data
    st.write('Presentation of Data and Preprocessing')
    st.write("Data Loaded Successfully", df.head())

# =====================================================================================
# CHOICE OF MODELS SECTION
# =====================================================================================

if page == pages[2]:
    st.write("Selecting the appropriate algorithm")
    image_url = "https://blog.medicalalgorithms.com/wp-content/uploads/2015/03/medical-algorithm-definition.jpg"
    st.image(image_url, caption="Machine learning, deep learning?", use_container_width=True)

    st.write("Since our project involves predicting a continuous target variable, Worldwide Harmonised Light vehicles Test Procedure (Ewltp (g/km)), this is inherently a regression problem.") 
    st.write("Our primary approach was to establish a robust baseline using multilinear regression, a widely accepted model for regression tasks.") 
    st.write("This choice allowed us to evaluate the model's performance under straightforward, interpretable assumptions about linear relationships between features and the target variable.") 


    st.write('**First models**')
    st.write("1- Linear Regression with Elastic Net")
    st.write("2- Decision Trees: chosen for their interpretability and ease of handling non-linear relationships. However, prone to overfitting.")
    st.write("3- Random Forests: as an ensemble method that aggregates multiple Decision Trees. They are more robust and show reduced tendency to overfit compared to a single Decision Tree.")
    st.write("4- XGBoost: also a tree-based ensemble method, which improves performance by sequentially building trees and learning from residual errors")
    st.write("5- Dense Neural Networks: lastly introduced as a deep learning approach to explore the possibility of capturing highly complex interactions among features that may not be adequately handled by tree-based algorithms")

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

# =====================================================================================
# MODELISATION AND ANALYSIS SECTION
# =====================================================================================

if page == pages[3]:
    st.write("Modelisation")

    # Check if df is loaded before accessing it
    if df is None:
        st.write("Data is not loaded. Please ensure data is available.")
    else:
        # Define file paths where the models are stored 
        xgboost_model_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\models\xgb_model.joblib'
        dnn_model_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\models\dnn_model.keras'
        lr_model_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\models\regression_model.joblib'

        # Function to load models using caching
        @st.cache_data
        def load_models():
            try:
                # Load models from the specified paths
                XG_model = joblib.load(xgboost_model_path)
                DNN_model = tf.keras.models.load_model(dnn_model_path)
                LR_model = joblib.load(lr_model_path)
                st.write("Models loaded successfully!")
            except Exception as e:
                st.write(f"Error loading models: {e}")
                return None, None, None
            return XG_model, DNN_model, LR_model

        # Load models (from cache if possible)
        XG_model, DNN_model, LR_model = load_models()

        # If models are not loaded, show message
        if XG_model is None or DNN_model is None or LR_model is None:
            st.write("Models not loaded. Please ensure they are saved and available in the given directory.")

        # Prepare the data (only need to preprocess once)
        target_column = 'Ewltp (g/km)'  # Target column
        X = df.drop(columns=[target_column])  # Features
        y = df[target_column]  # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Scaling the target variable as well (optional but can be helpful)
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

        # Step 1: Select model
        selected_model = st.selectbox(
            "Choose a model to evaluate",
            ["Linear Regression", "XGBoost", "Dense Neural Network"]
        )

        results = {}

        if selected_model == "Linear Regression":
            model = LR_model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test_scaled, y_pred)
            r2 = r2_score(y_test_scaled, y_pred)
            cv_r2 = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='r2').mean()
            results[selected_model] = {
                'Test MSE': mse,
                'Test R-squared': r2,
                'Cross-validated R-squared': cv_r2
            }

        elif selected_model == "XGBoost":
            model = XG_model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test_scaled, y_pred)
            r2 = r2_score(y_test_scaled, y_pred)
            cv_r2 = cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='r2').mean()
            results[selected_model] = {
                'Test MSE': mse,
                'Test R-squared': r2,
                'Cross-validated R-squared': cv_r2
            }

        elif selected_model == "Dense Neural Network":
            # For DNN, we do not need to train again, we simply use the already trained model
            y_pred = DNN_model.predict(X_test_scaled).flatten()
            mse = mean_squared_error(y_test_scaled, y_pred)
            r2 = r2_score(y_test_scaled, y_pred)
            cv_r2 = 'N/A'
            results[selected_model] = {
                'Test MSE': mse,
                'Test R-squared': r2,
                'Cross-validated R-squared': cv_r2
            }

        # Display results for the selected model
        st.write(f"Results for {selected_model}:")
        st.write(results[selected_model])

        # Show comparison table for all models (if checkbox is selected)
        if st.checkbox('Show all models comparison'):
            if not results:  # If results are empty (no models selected yet)
                st.write("Please select a model to see results.")
            else:
                comparison_df = pd.DataFrame(results).T
                st.write(comparison_df)


# =====================================================================================
# Conclusion
# =====================================================================================


if page == pages[4]:
    st.write("Modelisation")






