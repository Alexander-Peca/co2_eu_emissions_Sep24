# =====================================================================================
# LIBRARIES SECTION
# =====================================================================================

import streamlit as st
import gdown  # to load data from Tillmann's google drive
import time
import joblib  # For saving and loading models
import os
import requests
from io import BytesIO
from PIL import Image

# Import standard libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Import libraries for the modeling
from sklearn.model_selection import train_test_split
#from sklearn.model_selection cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import ElasticNetCV
# import xgboost as xgb

# TensorFlow for DNN
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping


# =====================================================================================
# STREAMLIT OVERALL STRUCTURE
# =====================================================================================

st.title("CO2 emissions by vehicles")
pages = ["Home", "Data and Preprocessing", "Choice of Models", "Modelisation and Analysis", "Conclusion"]
page = st.sidebar.radio("Go to", pages)

# =====================================================================================
# HOME SECTION
# =====================================================================================

if page == pages [0]:
    st.write('Context')

    image_url = "https://www.mcilvainmotors.com/wp-content/uploads/2024/02/Luxury-Car-Increased-Emission.jpg"
    st.image(image_url, caption="Luxury Car Emissions", use_container_width=True)
    
    # Display the text
    st.write("Identifying the vehicles that emit the most CO2 is important to identify the technical characteristics that play a role in pollution.") 
    st.write("Predicting this pollution in advance makes it possible to prevent the appearance of new types of vehicles (new series of cars for example.")

    st.write('**Project Purpose and Goals**')
    st.write("This project focuses on using machine learning to help the automotive industry meet the EU's 2035 target of 0 g CO₂/km for passenger cars and vans. By analyzing extensive vehicle data, machine learning models can identify factors influencing CO₂ emissions, aiding manufacturers in designing low-emission vehicles. This ensures compliance with regulations, reduces penalties, and enhances brand reputation. The project also aims to optimize production strategies by enabling early design adjustments, especially as the industry shifts towards zero-emission vehicles and considers alternative energy sources like hydrogen or electricity for appliances in motorhomes (as a collateral effect)")



# =====================================================================================
# DATA PREPROCESSING SECTION
# =====================================================================================


# # Google drive file link of final preprocessed data set
# file_url = "https://drive.google.com/uc?id=13hNrvvMgxoxhaA9xM4gmnSQc1U2Ia4i0"  # file link to Tillmann's drive

# # Output filename where the file will be saved
# output = 'data.parquet'

# try:
#     # Try downloading the file from Google Drive
#     gdown.download(file_url, output, quiet=False)

#     # Load the data into a DataFrame
#     df = pd.read_parquet(output)
#     st.write("Data loaded successfully from Google Drive")

# except Exception as e:
#     # If Google Drive download fails, load data from local path
#     st.write("Failed to load data from Google Drive. Reverting to local data.")
#     df = pd.read_parquet(r'C:\Users\alexa\Downloads\ProjectCO2--no Github\Data\minimal_withoutfc_dupesdropped_frequencies_area_removedskew_outliers3_0_NoW_tn20_mcp00.10.parquet')


# Define the Google Drive file URL
file_id = "13hNrvvMgxoxhaA9xM4gmnSQc1U2Ia4i0"
file_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Initialize an empty DataFrame if it doesn't already exist
if 'df' not in st.session_state:
    st.write("..df not in session..")
    st.session_state.df = None
else:
    st.write("..df in session..")
    
# Load the data only if it's not already loaded in the session
if st.session_state.df is None:
    try:
        st.write("Loading data from Google Drive...")
        
        # Request the file from Google Drive
        response = requests.get(file_url)
        response.raise_for_status()  # Ensure the request was successful

        # Load the data directly into a DataFrame
        st.session_state.df = pd.read_parquet(BytesIO(response.content))
        df = st.session_state.df
        st.write("Data loaded successfully from Google Drive")

    except Exception as e:
        st.write("Failed to load data from Google Drive. Reverting to local data.")
        # df = pd.read_parquet(r'C:\Users\alexa\Downloads\ProjectCO2--no Github\Data\minimal_withoutfc_dupesdropped_frequencies_area_removedskew_outliers3_0_NoW_tn20_mcp00.10.parquet')


else:
    st.write("Data is already loaded, using cached DataFrame.")
    df = st.session_state.df  # Create local reference if already cached





# Display the data
if page == pages[1]:
    st.write('Presentation of Data and Preprocessing')
    st.write("Data Loaded Successfully", df.head())





# =====================================================================================
# CHOICE OF MODELS SECTION
# =====================================================================================

if page == pages[2]:
    st.write("Selecting the appropiate algorithm")
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
    st.markdown("""
                - Redundancy with XGBoost: it's advanced algorithm that surpasses the performance of Decision Trees and Random Forests.
                - Bullet point 2: In our tests, XGBoost not only yielded higher accuracy but also demonstrated more stable performance across various data splits.
                - Bullet point 3: XGBoost is optimized for scalability with large datasets and offers greater control over hyperparameters, making it better suited for fine-tuning (Literature). 
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

    # Google Drive image URLs
    linear_regression_image_url = "https://drive.google.com/uc?export=view&id=1JWR6BqH8eebiZmtyLOgslDDGHGOq4ec3"
    xgboost_image_url = "https://drive.google.com/uc?export=view&id=14iFU17b6_wMzsYNTdtda9ZsOrbp0Uq7D"

    LinReg_img_resp = requests.get(linear_regression_image_url)
    LinReg_img = Image.open(BytesIO(LinReg_img_resp.content))

    XGB_img_resp = requests.get(xgboost_image_url)
    XGB_img = Image.open(BytesIO(XGB_img_resp.content))

    # Show images in Streamlit without downloading
    st.image(LinReg_img, caption="Feature Importance - Linear Regression", use_container_width=True)
    st.image(XGB_img, caption="SHAP Values - XGBoost", use_container_width=True)

    



# =====================================================================================
# MODELISATION AND ANALYSIS SECTION - ONLY THIS PAGE SHOWS THE MODELS AND RESULTS
# =====================================================================================

if page == pages[3]:
    st.write("Modelisation")

    # Define file paths where the models are stored
    # load_dir = r'C:\Users\alexa\Downloads\ProjectCO2--no Github\Presentation streamlit'
    # load_dir = r"C:\Users\nx10\DS-Project\trained_Models"

    # Build a path relative to this script's directory
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # load_dir = os.path.normpath(os.path.join(current_dir, "../models"))
    # xgboost_model_path = os.path.join(load_dir, 'xgb_model.joblib')
    # xgboost_model_path = "C:\\Users\\nx10\\DS-Project\\trained_Models\\xgboost_model.joblib"
    # st.write(xgboost_model_path)

    
    # Function to load models using caching
    @st.cache_data
    def load_model_from_gdrive(model_url):
    
        try:
            # st.write(f"Downloading and loading the model from {model_url}...")
            
            # Download the model content without authentication
            response = requests.get(model_url)
            response.raise_for_status()  # Check if the request was successful
            
            # Load the model from the response content
            model = joblib.load(BytesIO(response.content))
            st.write("Model loaded successfully from Google Drive.")
            return model
            
        except requests.exceptions.HTTPError as e:
            st.write("Failed to load the model from Google Drive.")
            st.write(f"HTTP Error: {e}")
        except Exception as e:
            st.write("An error occurred while loading the model.")
            st.write(e)
            return None

    # models URLs
    XG_model_url = "https://drive.google.com/uc?id=171P5hEOH5HWQ7n0qIM5ahlMl4yG4NlFe"  #joblib
    LR_model_url =  "https://drive.google.com/uc?id=1gY8JymL1UBOrRjr4GDuvIxX4ZOFn_X4o"  #joblib
    # DNN_model_url = "https://drive.google.com/uc?id=1K65XtPYgXJ0wkxBm2a0HJiQRL0cIz_i9"  #joblib
    # DNN_model_url = "https://drive.google.com/uc?id=1ZqiNUwoRJ47I4vNuU-Fj6q15hJn3L0wT"  #keras

    # load models
    XG_model = load_model_from_gdrive(XG_model_url)
    LR_model =  load_model_from_gdrive(LR_model_url)
    #DNN_model = load_model_from_gdrive(DNN_model_url)
    DNN_model = None



    
    def load_models():
        try:
            # Check if models are loaded from cache
            # XG_model = joblib.load(xgboost_model_path)
            # DNN_model = tf.keras.models.load_model(dnn_model_path)
            #LR_model = joblib.load(lr_model_path)
            st.write("Models loaded successfully!")
        except:
            st.write("Models not found. Please ensure they are saved and available in the given directory.")
            return None, None, None
        return XG_model, DNN_model, LR_model

    # Load models (from cache if possible)
    # XG_model, DNN_model, LR_model = load_models()

    # If models are not loaded, show message
    if XG_model is None or DNN_model is None or LR_model is None:
            st.write("Models not loaded. Please ensure they are saved and available in the given directory.")
            if XG_model is not None:
                st.write("XG_model loaded")
            if LR_model is not None:
                st.write("LR_model loaded")
            if DNN_model is not None:
                st.write("DNN_model loaded")


    # Prepare the data
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

    # Select model
    selected_model = st.selectbox(
        "Choose a model to evaluate",
        ["Linear Regression", "XGBoost", "Dense Neural Network"]
    )

    results = {}

    if selected_model == "Linear Regression":
        model = LR_model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred) # changing y_test_sclaled to y_test
        r2 = r2_score(y_test, y_pred)            # changing y_test_sclaled to y_test
        cv_r2 = None #cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='r2').mean()
        results[selected_model] = {
            'Test MSE': mse,
            'Test R-squared': r2,
            'Cross-validated R-squared': cv_r2
        }

    elif selected_model == "XGBoost":
        model = XG_model
        y_pred = model.predict(X_test_scaled)
        mse = float(mean_squared_error(y_test, y_pred))   # changing y_test_sclaled to y_test + type: float
        r2 = r2_score(y_test, y_pred)                     # changing y_test_sclaled to y_test
        cv_r2 = None # cross_val_score(model, X_train_scaled, y_train_scaled, cv=5, scoring='r2').mean()
        results[selected_model] = {
            'Test MSE': mse,
            'Test R-squared': r2,
            'Cross-validated R-squared': cv_r2
        }

    #elif selected_model == "Dense Neural Network":
        # DNN_model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=16, callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)], verbose=1)
        # No need to retrain/fit the model, snce trained model was loaded
        # y_pred = DNN_model.predict(X_test_scaled).flatten()
        # mse = mean_squared_error(y_test_scaled, y_pred)
        # r2 = r2_score(y_test_scaled, y_pred)
        # cv_r2 = None
        # results[selected_model] = {
        #     'Test MSE': mse,
        #     'Test R-squared': r2,
        #     'Cross-validated R-squared': cv_r2
        #}

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

    # Add section for interpretability examples
    st.write('**Examples of Interpretability**')

    # Define paths to images
    # linear_regression_image_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\visualization\Feature Importance Linear Regression.png'
    # xgboost_image_path = r'C:\Users\alexa\Downloads\aug24_bds_int---co2\src\visualization\Feature Importance XG Boost.png'

    # Display the images in Streamlit
    # st.image(linear_regression_image_path, caption='Feature Importance - Linear Regression', use_column_width=True)
    # st.image(xgboost_image_path, caption='SHAP Values - XGBoost', use_column_width=True)



# =====================================================================================
# Conclusion
# =====================================================================================


if page == pages[4]:
    st.write("Modelisation")






