# =====================================================================================
# LIBRARIES SECTION
# =====================================================================================

import streamlit as st
import gdown  # to load data from Tillmann's google drive
import os
import sys
import requests
from io import BytesIO

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


st.markdown(
    '<p style="font-size: 20px; font-style: italic; margin-bottom: -50px;">Project defence</p>', 
    unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 48px;'>CO<sub>2</sub> emissions by vehicles</h1>", unsafe_allow_html=True)  # Groß mit CO₂

pages = ["Home", "Data and Preprocessing", "Choice of Models", "Evaluating Model Performance", "Conclusions"]
page = st.sidebar.radio("Go to", pages)

# =====================================================================================
# HOME SECTION
# =====================================================================================

if page == pages[0]:
    # "Click 1" checkbox
    show_click_1 = st.checkbox("Introduction")

    # Conditional image display
    if show_click_1:
        # Display the second image
        image_url = "https://www.researchgate.net/profile/Yan-Ma-14/publication/344081717/figure/fig1/AS:953388120420352@1604316849199/CO2-emissions-in-the-European-Union-EU-a-evolution-of-CO2-emission-in-the-EU-by.jpg"
        #https://www.europarl.europa.eu/resources/library/images/20220524PHT31019/20220524PHT31019_original.jpg"
        st.image(image_url, caption="Comparison of emissions by sector- Dissertation by Yan Ma 2019", use_container_width=True)

            # Project Purpose
        st.write('## Project Purpose')
        st.markdown("""
        - Car emissions remain the highest compared to other sectors in the EU.
        - Using ML and data science to analyze emission factors.
        - Pinpointing technical characteristics of cars which contribute to pollution.
        """)
    else:
        # Display the first image
        image_url = "https://www.mcilvainmotors.com/wp-content/uploads/2024/02/Luxury-Car-Increased-Emission.jpg"
        st.image(image_url, caption="Representative image - emissions from a car", use_container_width=True)


    # Checkbox for Economic Benefits
    show_benefits = st.checkbox("Benefits & Goals")
    if show_benefits:
        st.header("Economic Selling Points")
        st.markdown("""
        - Preventing the introduction & development of high-pollution vehicle types.  
        ➡️ Avoiding penalties for non-compliance with CO₂ regulations.
        - Reducing redesign costs by optimizing vehicle development.  
        ➡️ Efficient production strategies.  
        ➡️ Enabling competitive pricing.   
        - Boosting brand reputation with eco-friendly recognition.
        """)

        st.header("Technical Goals & Scientific Contribution")
        st.markdown("""
        - Data mining, engenering, select features, train, and evaluate models.
        - Identifying emission-related data patterns.  
        - Focus on CO₂ emissions during operation.
        - Correlate vehicle features with CO₂ emissions.
        - Predict emissions from technical characteristics.
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

        - **One-hot encoded** categorical variables with **baseline categories dropped** to prevent multicollinearity.
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
            - **'At2 (mm)'**: Replaced **1,411** outliers with median **1,542.0**.

        - **Non-Gaussian Columns (Capped outliers):**
            - **'Ewltp (g/km)'**: Capped **151,536** outliers between **33.0** and **243.0**.

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
    st.image(xgboost_image_path, caption="SHAP Values - XGBoost", use_container_width=False, width = 450)
    st.image(dnn_image_path, caption="Weights First Layer - DNN", use_container_width=True)




# =====================================================================================
# MODELISATION AND ANALYSIS SECTION
# =====================================================================================

if page == pages[3]:
    st.write("## Models Performance")

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
    st.write("## Conclusions")

    # Conclusion about the Models
    st.subheader("Model Performance")
    st.write("""
    **XGBoost**:
    - Achieved the best predictive accuracy.
    - Lowest Mean Squared Error (MSE) and highest R-squared.
    - Effectively captures non-linear relationships.
    - Provides an optimal balance of performance and efficiency.
    
    **Dense Neural Network (DNN)**:
    - A strong contender with fast training and high R-squared.
    - Requires further fine-tuning to prevent overfitting.

    **Linear Regression**:
    - Offers simplicity and interpretability.
    - Underperforms relative to XGBoost and DNN.

    **Recommendation**: XGBoost 
    """)

    # Checkbox for "Outlook and Improvements"
    show_outlook = st.checkbox("Outlook and Improvements")

    if show_outlook:
        st.write("""
        - **Ensemble Learning**: Combining Linear Regression, XGBoost, and Dense Neural Networks for improved predictions.
        - **Hyperparameter Tuning**: Leveraging techniques like Bayesian Optimization for XGBoost and DNN.
        - **Advanced Architectures**: Experimenting with CNNs or Residual Connections to boost model performance.
        - **Data Augmentation**: Using methods like SMOTE to address class imbalances.
        """)

