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

# Specify Google Drive URLs for the images
target_vars_image_url = "https://drive.google.com/uc?id=1JRV_WK7EmEuOvktEQnnLUUM6zTEgXI_x"
# Specify Path(name) for the images
target_vars_image_path = "target_vars_all_years.png"
# Download images
gdown.download(target_vars_image_url, target_vars_image_path, quiet=False)


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
    st.write('## Overview of Data and Preprocessing')

    st.write("Our primary dataset is the **CO₂ emissions from new passenger cars** provided by the European Environment Agency (EEA). The initial dataset was over **16 GB** on disc and included over **14 million rows** and **41 columns**.")

    st.write("### Distribution of Target Variables")
    # Show image
    st.image(target_vars_image_path, use_container_width=True)


    #####################################################################
    #                    Data Preprocessing Steps                       #
    #####################################################################

    st.write("### Data Preprocessing Steps")
        
    with st.expander("**1. Data Loading and Initial Processing**"):
        st.write("""
        - Data from **2010 to 2023** was downloaded in CSV format.
        - Due to its size, data was processed on a per-year basis.
        - Memory optimization was performed by specifying data types and downcasting numerical columns.
        - Duplicate entries were consolidated by counting their frequencies, reducing the dataset size to approximately **300 MB** on disk and **1.6 GB** in memory.
        """)

    with st.expander("**2. Data Cleaning**"):
        st.write("""
        - Cleaned categorical variables with inconsistent categories due to misspellings or variations using mapping dictionaries.
        - Columns cleaned include **'Country'**, **'Ct'**, **'Cr'**, **'Fm'**, **'Ft'**, **'Mp'**, **'Mh'**, and **'IT'**.
        - Parsed the **'IT'** variable, representing **Innovative Technologies**, to extract individual codes.
        """)

    with st.expander("**3. Major Dataset Selection Decisions**"):
        st.write("""
        - Selected **'Ewltp (g/km)'** as the target variable and removed **'Enedc (g/km)'** (obsolete testing standard).
        - **Dropped all years prior to 2019**, focusing on the most recent data.
        - Selected the top three countries by frequency: **Germany (DE)**, **France (FR)**, and **Italy (IT)**.
        - Excluded electric and plug-in hybrid cars to focus on combustion engine vehicles.
        """)

    with st.expander("**4. Feature Selection and Dropping Columns**"):
        st.write("""
        We focused on retaining technical attributes relevant for modeling CO₂ emissions and dropped non-essential or redundant columns.

        **Columns Dropped:**

        - **Car Identifiers**: 'Ve', 'Va', 'T', 'Tan', 'ID'
        - **Administrative Values**: 'Date of registration', 'Status', 'Vf', 'De'
        - **Data Related Columns**: 'r'
        - **Brand Related**: 'Mp', 'Man', 'MMS'
        - **Temporary or Transformed**: 'IT', 'IT_valid', 'IT_invalid', 'Non_Electric_Car'
        - **Target Redundant**: 'Enedc (g/km)', 'Fuel consumption', 'Erwltp (g/km)', 'Ernedc (g/km)'
        - **Collinear Attributes**: 'm (kg)'
        - **Other Columns**: 'Cr', 'ech', 'RLFI', 'Electric range (km)', 'z (Wh/km)', 'year'
        - **Individual Columns**: 'Mk', 'Country', 'Cn', 'VFN'

        **Columns Retained:**

        - **Categorical Attributes**:
            - **'Mh'**: Manufacturer name EU standard denomination
            - **'Ct'**: Category of Vehicle (Passenger cars vs. off-road)
            - **'Ft'**: Fuel type (Petrol, Diesel, LPG, etc.)
            - **'Fm'**: Fuel mode (Mono-Fuel, Bi-Fuel, etc.)
            - **'IT_1' to 'IT_5'**: Innovative Technologies codes

        - **Numerical Attributes**:
            - **'Mt'**: Test mass in kg as measured for the WLTP test
            - **'ep (KW)'**: Engine power in kW
            - **'ec (cm3)'**: Engine capacity in cm³
            - **'At1 (mm)'**: Axle width (steering axle) in mm
            - **'Area (m²)'**: Calculated from existing dimensions
        """)

    with st.expander("**5. Category Selection and Encoding**"):
        st.write("""
        We refined categorical variables to focus on the most significant categories.
        
        **General Selection Criteria:**
        - Categories were selected using two parameters:
            - **top_n = 20**: Retained the top 20 most frequent categories.
            - **min_cat_percent = 0.1**: Ensured each retained category represents at least 0.1% of the total dataset.
        - Categories not meeting these criteria were labeled as **'Other'**.

        **Category Selection Details:**
        - **'Mh' (Manufacturer)**:
            - Kept categories: 'VOLKSWAGEN', 'BMW AG', 'MERCEDES-BENZ AG', 'AUDI AG', 'SKODA', 'FORD WERKE GMBH', 'SEAT', 'RENAULT', 'PSA', 'OPEL AUTOMOBILE', 'AUTOMOBILES PEUGEOT', 'VOLVO', 'PORSCHE', 'JAGUAR LAND ROVER LIMITED', 'FIAT GROUP', 'AUTOMOBILES CITROEN', 'AA-IVA', 'STELLANTIS EUROPE', 'TOYOTA', 'DACIA'
            - Replaced **200,589** values with 'Other'.

        - **'Ct' (Vehicle Category)**:
            - Kept categories: 'M1', 'M1G'
            - Replaced **594** values with 'Other'.

        - **'Ft' (Fuel Type)**:
            - Kept categories: 'DIESEL', 'PETROL', 'NG-BIOMETHANE', 'LPG', 'E85'
            - **Dropped 26,963 rows** not in specified categories.

        - **'Fm' (Fuel Mode)**:
            - Kept categories: 'M', 'H', 'B'
            - Replaced **1,034** values with 'Other'.

        - **'IT' (Innovative Technologies)**:
            - Retained top 20 IT codes occurring more than 0.1%; others labeled as 'Other'.

        **Encoding:**

        - One-hot encoded categorical variables with baseline categories dropped to prevent multicollinearity.
        - **'IT'** codes were one-hot encoded across **'IT_1'** to **'IT_5'** columns.

        """)

    with st.expander("**6. Handling Outliers**"):
        st.write("""
        **Outlier Handling:**

        - Applied an IQR multiplier of **3.0** for less aggressive outlier removal.
        - **Gaussian Columns (Replaced outliers with median):**
            - **'Mt'**: Replaced **3,131** outliers with median **1,500.0**.
            - **'W (mm)'**: Replaced **143,449** outliers with median **2,624.0**.
            - **'At1 (mm)'**: Replaced **2,519** outliers with median **1,545.0**.
            - **'At2 (mm)'**: Replaced **1,403** outliers with median **1,542.0**.

        - **Non-Gaussian Columns (Capped outliers):**
            - **'Ewltp (g/km)'**: Capped **151,522** outliers between **33.0** and **243.0**.

        - Highly skewed attributes **'ep (KW)'** and **'ec (cm3)'** were transformed using the **Box-Cox** method.

        """)

    with st.expander("**7. Handling Missing Values**"):
        st.write("""
        **Missing Values Handling:**

        - Dropped rows with missing values in key columns:
            - **'Mh'**: Dropped **13** rows.
            - **'Ct'**: Dropped **2,095** rows.
            - **'Mt'**: Dropped **73,424** rows.
            - **'W (mm)'**: Dropped **742,035** rows.
            - **'At1 (mm)'**: Dropped **868,257** rows.
            - **'At2 (mm)'**: Dropped **869,004** rows.
            - **'Ft'**: Dropped **4** rows.
            - **'Fm'**: Dropped **3** rows.

        - Left NaNs in **'IT'** columns as missing values are expected.

        """)

    with st.expander("**8. Feature Engineering**"):
        st.write("""
        - **Created new feature 'Area (m²)'**:
            - Calculated as: **Area = W * (At1 + At2) / 2 / 1,000,000**
            - Represents the car's footprint, capturing size-related characteristics.
        - Removed **'W (mm)'** and **'At2 (mm)'** to reduce collinearity.

        """)
        
    with st.expander("**9. Duplicate Removal**"):
        st.write("""
        - Removed duplicate rows and recorded frequencies to maintain data representation.
        - Initial row count: **3,731,632**
        - Final row count after removing duplicates: **2,000,450**
        """)

    

    #####################################################################
    #                         Final Dataset                             #
    #####################################################################
    
    st.write("### **Final Dataset:**")
    st.write("""- The final dataset contains **2,000,450 rows** and **56 columns**, reduced from the initial **16 GB** to approximately **19 MB** on disk and **282.4 MB** in memory.
    """)

    st.write("### Numerical and Categorical Attribute Distributions")

      # Get sorted column names
    columns = sort_columns(df)

    with st.expander("Show all Columns in DataSet"):
        columns_str = ", ".join(columns)
        st.write(columns_str)

    # List of tuples with prefixes, baseline categories, and long attribute names
    categorical_info = [
        ('Ct', 'Ct_M1 (Baseline)', 'Vehicle Type'),
        ('Fm', 'Fm_B (Baseline)', 'Fuel Mode'),
        ('Ft', 'Ft_DIESEL (Baseline)', 'Fuel Type'),
        ('Mh', 'Mh_AA-IVA (Baseline)', 'Manufacturer'),
        ('IT_code', 'IT_code_None (Baseline)', 'Innovative Technologies')
    ]

    numerical_summary, categorical_summaries = create_attribute_summary_tables(df, categorical_info)

    # Descriptions for each prefix
    descriptions = {
        'Ct': """
        - **M1**: Passenger cars (up to 8 seats + driver).
        - **M1G**: Off-road passenger cars.
        """,
        'Fm': """
        - **M**: Mono-Fuel (Petrol, Diesel, LNG, etc.).
        - **B**: Bi-Fuel vehicles (e.g., LNG and Petrol).
        - **H**: Non-plugin Hybrids.
        """,
        'Ft': """
        - **DIESEL**: Diesel fuel.
        - **PETROL**: Petrol fuel.
        - **E85**: 85% ethanol, 15% petrol.
        - **LPG**: Liquefied petroleum gas.
        - **NG-BIOMETHANE**: Natural gas or biomethane.
        """,
        'Mh': """
        - **Mh_XXX**: Standardized EU manufacturer names.
        - **Mh_AA-IVA**: Individual vehicle approvals (non-standard).
        """,
        'IT_code': """
        - **None**: No approved innovative technology.
        - **IT_code_e1 2**: Alternator.
        - **IT_code_e1 29**: Alternator.
        - **IT_code_e13 17**: Alternator.
        - **IT_code_e13 19**: LED Lights.
        - **IT_code_e13 28**: LED Lights.
        - **IT_code_e13 29**: Alternator.
        - **IT_code_e13 37**: LED Lights.
        - **IT_code_e2 17**: Alternator.
        - **IT_code_e2 29**: Alternator.
        - **IT_code_e24 17**: Alternator.
        - **IT_code_e24 19**: LED Lights.
        - **IT_code_e24 28**: 48V Motor Generators.
        - **IT_code_e24 29**: Alternator.
        - **IT_code_e24 3**: Engine compartment encapsulation system.
        - **IT_code_e24 37**: LED Lights.
        - **IT_code_e8 19**: LED Lights.
        - **IT_code_e8 29**: Alternator.
        - **IT_code_e8 37**: LED Lights.
        - **IT_code_e9 29**: Alternator.
        - **IT_code_e9 37**: LED Lights.
        """
    }

    # Streamlit Output
    st.write("#### Numerical Attributes")
    st.dataframe(numerical_summary)

    st.write("#### Categorical Attributes")
    for prefix, summary_table in categorical_summaries.items():
        long_name = [entry[2] for entry in categorical_info if entry[0] == prefix][0]
        st.markdown(f"**{prefix} ({long_name})**")
        with st.expander(f"Show details for {long_name}"):
            if prefix in descriptions:
                st.markdown(descriptions[prefix])
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
    | Dense Neural Network    | No                       | Not applicable, Ridge regularization | No (validation set used) | Weights First Layer         |
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

