# Importing necessary librarys
# import numpy as np
import pandas as pd
import numpy as np
import pyarrow.parquet as pq  # For working with parquet files
import re

import sys
import os

# Path to the neighboring 'data' folder in the local repository
data_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))


def inspect_data(df):
    """
    Function to perform an initial data inspection on a given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.

    Returns:
    None
    """
    print("="*40)
    print("🚀 Basic Data Overview")
    print("="*40)

    # Print the shape of the DataFrame (rows, columns)
    print(f"🗂 Shape of the DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # Display the first 5 rows of the dataset
    print("\n🔍 First 5 rows of the DataFrame:")
    print(df.head(5))

    # Get information about data types, missing values, and memory usage
    print("\n📊 DataFrame Information:")
    df.info()

    # Show basic statistics for numeric columns
    print("\n📈 Summary Statistics of Numeric Columns:")
    print(df.describe())

    # Count missing values and display
    print("\n❓ Missing Values Overview:")
    missing_df = count_missing_values(df)
    print(missing_df)

    # Print unique values for categorical and object columns
    print("\n🔑 Unique Values in Categorical/Object Columns:")
    print_unique_values(df)

def count_missing_values(df):
    """
    Function to count missing values and provide an overview of each column.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.

    Returns:
    pd.DataFrame: DataFrame with missing value counts, data types, and percentages.
    """
    missing_counts = {}

    for col in df.columns:
        # Count missing values (NaN)
        missing_count = df[col].isna().sum()

        # Store the results
        missing_counts[col] = {
            'Dtype': df[col].dtype.name,
            'Missing Count': missing_count,
            'Percent Missing': f"{missing_count / len(df) * 100:.2f}%"
        }

    # Convert the results to a DataFrame for easier viewing
    result_df = pd.DataFrame(missing_counts).T
    result_df = result_df[['Dtype', 'Missing Count', 'Percent Missing']]

    return result_df

def print_unique_values(df):
    """
    Function to print the number of unique values for categorical and object columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.

    Returns:
    None
    """
    cat_obj_cols = df.select_dtypes(include=['category', 'object']).columns
    unique_counts = {}

    for col in cat_obj_cols:
        unique_counts[col] = df[col].nunique()

    if unique_counts:
        for col, count in unique_counts.items():
            print(f"Column '{col}' has {count} unique values")
    else:
        print("No categorical or object columns found.")
        
        
def count_missing_values2(df, columns):
    """
    Counts occurrences of "nan", "None", np.nan, None, pd.NA, and '' (empty string) in the specified columns of a DataFrame.
    Also includes the data type of each column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the columns to check.
    columns (list): List of column names (as strings) to check for missing values.

    Returns:
    pd.DataFrame: A DataFrame summarizing the count of missing value types per column.
    """
    results = []
    for col in columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            # If the column is categorical, convert it to object to capture all kinds of missing values
            col_data = df[col].astype(object)
        else:
            col_data = df[col]

        # Total missing (both np.nan, pd.NA, None)
        total_missing = col_data.isna().sum()

        # Count only np.nan by checking where it's actually a float and is NaN
        np_nan_count = (col_data.apply(lambda x: isinstance(x, float) and pd.isna(x))).sum()

        # Count actual `None` values (None treated as an object)
        none_object_count = (col_data.apply(lambda x: x is None)).sum()

        # pd.NA count: in categorical columns, we check if the missing value type is distinct from np.nan
        if pd.api.types.is_categorical_dtype(df[col]):
            pd_na_count = col_data.isna().sum() - none_object_count - np_nan_count
        else:
            pd_na_count = total_missing - np_nan_count

        counts = {
            'Column': col,
            'Data Type': df[col].dtype,
            'nan (string)': (col_data == 'nan').sum(),
            'None (string)': (col_data == 'None').sum(),
            'np.nan': np_nan_count,
            'pd.NA': pd_na_count,
            'None (object)': none_object_count,
            'Empty string': (col_data == '').sum(),
        }
        results.append(counts)

    return pd.DataFrame(results)

        
# Function to rename categorical values using mappings
def rename_catval(df, attribute, mappings):
    """
    Rename categorical values in a DataFrame column based on provided mappings.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to be renamed.
    attribute (str): The name of the column to be renamed.
    mappings (list of tuples): Each tuple contains a list of aliases and the target name.
                               Example: [ (['old_name1', 'old_name2'], 'new_name'), ... ]
    """
    # Convert the column to a non-categorical type (e.g., string)
    df[attribute] = df[attribute].astype('string')

    # Build a rename dictionary from the mappings
    rename_dict = {}
    for aliases, target_name in mappings:
        for alias in aliases:
            rename_dict[alias] = target_name

    # Replace values
    df[attribute] = df[attribute].replace(rename_dict)

    # Convert the column back to categorical
    df[attribute] = df[attribute].astype('category')
    
    

# restore NaNs and turn to "category" specified columns    
def all_to_nan_and_cat(df, cols):
    for col in cols:
        df[col] = df[col].replace(['nan', 'None'], np.nan).fillna(np.nan)
        df[col] = df[col].astype('category')
        
        
        
# Filter categories        
def filter_categories(df, column, drop=False, top_n=None, categories_to_keep=None, other_label='Other'):
    """
    Filter categories in a column based on top_n or an explicit list of categories to keep.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical column.
    - column (str): The name of the categorical column.
    - drop (bool, optional): If True, drop rows with categories not in categories_to_keep or top_n.
                             If False, label them as 'Other'. Defaults to False.
    - top_n (int, optional): Number of top categories to keep based on frequency.
    - categories_to_keep (list, optional): List of categories to retain.
    - other_label (str, optional): Label for aggregated other categories. Defaults to 'Other'.
    
    Returns:
    - pd.DataFrame: DataFrame with updated categorical column.
    
    Notes:
    - If both top_n and categories_to_keep are provided, categories_to_keep will be ignored, and only top_n will be used.
    
    Raises:
    - ValueError: If neither top_n nor categories_to_keep is provided.
    """
    initial_row_count = len(df)
    
    if top_n is not None:
        # Ignore categories_to_keep if top_n is provided
        top_categories = df[column].value_counts().nlargest(top_n).index
        if drop:
            df = df[df[column].isin(top_categories) | df[column].isna()]
            rows_dropped = initial_row_count - len(df)
            print(f"Dropped {rows_dropped} rows where '{column}' is not in the top {top_n} categories.")
        else:
            if pd.api.types.is_categorical_dtype(df[column]):
                # Add 'Other' to categories if not present
                if other_label not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([other_label])
            mask = ~(df[column].isin(top_categories) | df[column].isna())
            num_replaced = mask.sum()
            df.loc[mask, column] = other_label
            print(f"Replaced {num_replaced} values in '{column}' with '{other_label}' where not in top {top_n} categories.")
    elif categories_to_keep is not None:
        if drop:
            df = df[df[column].isin(categories_to_keep) | df[column].isna()]
            rows_dropped = initial_row_count - len(df)
            print(f"Dropped {rows_dropped} rows where '{column}' is not in the specified categories to keep.")
        else:
            if pd.api.types.is_categorical_dtype(df[column]):
                # Add 'Other' to categories if not present
                if other_label not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([other_label])
            mask = ~(df[column].isin(categories_to_keep) | df[column].isna())
            num_replaced = mask.sum()
            df.loc[mask, column] = other_label
            print(f"Replaced {num_replaced} values in '{column}' with '{other_label}' where not in specified categories to keep.")
    else:
        raise ValueError("Either top_n or categories_to_keep must be provided.")
    
    return df





# Retain top_n ITs 
import pandas as pd

def retain_top_n_ITs(df, top_n, IT_columns=['IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5'], other_label='Other'):
    """
    Retain top_n ITs (innovative technology codes) and replace others with `other_label`.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the IT columns.
    - top_n (int): Number of top ITs to retain based on frequency.
    - IT_columns (list, optional): List of IT column names. Defaults to ['IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5'].
    - other_label (str, optional): Label for aggregated other ITs. Defaults to 'Other'.
    
    Returns:
    - pd.DataFrame: DataFrame with updated IT columns.
    """
    print(f"Retaining top {top_n} ITs and labeling others as '{other_label}'...")
    
    # Identify which IT columns are present in the DataFrame
    existing_IT_columns = [col for col in IT_columns if col in df.columns]
    missing_IT_columns = [col for col in IT_columns if col not in df.columns]
    
    if missing_IT_columns:
        print(f"Warning: The following IT columns are not in the DataFrame and will be skipped: {missing_IT_columns}")
    
    if not existing_IT_columns:
        print("No valid IT columns found for processing. Returning original DataFrame.")
        return df.copy()
    
    # Concatenate all existing IT columns to compute global top_n
    combined_ITs = pd.concat([df[col] for col in existing_IT_columns], axis=0, ignore_index=True).dropna()
    
    # Determine the top_n ITs
    top_n_ITs = combined_ITs.value_counts().nlargest(top_n).index.tolist()
    print(f"Top {top_n} ITs: {top_n_ITs}")
    
    # Replace ITs not in top_n with other_label using vectorized operations
    for col in existing_IT_columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            # Add 'Other' to categories if not present
            if other_label not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([other_label])
        
        # Replace ITs not in top_n_ITs with other_label
        original_unique = df[col].dropna().unique().tolist()
        df[col] = df[col].where(df[col].isin(top_n_ITs), other_label)
        updated_unique = df[col].dropna().unique().tolist()
        
        print(f"Updated '{col}' categories: {updated_unique}")
    
    print("Replacement complete.\n")
    return df.copy()





# Generate categories lists and functions to choose
def generate_category_lists(df, max_categories=20):
    """
    Generates summaries of categories for specified columns and provides
    pre-filled filter_categories and retain_top_n_ITs function calls for easy integration.

    The output is formatted with comments and code snippets that can be
    directly copied into your codebase. You only need to manually delete
    the category names you want to exclude from the `categories_to_keep` list or adjust the `top_n` parameter.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical columns.
    - max_categories (int, default=20): Maximum number of categories to display 
      based on value counts for columns with high cardinality.

    Returns:
    - None: Prints the summaries and function calls to the console.
    """
    # Define the regular categorical columns to process
    # Identify categorical columns, excluding those starting with "IT_"
    categorical_columns = [
        col for col in df.select_dtypes(include=['category', 'object']).columns
        if not col.startswith("IT_")
    ]
    
    for col in categorical_columns:
        if col not in df.columns:
            print(f"# Column '{col}' not found in DataFrame.\n")
            continue
        
        # Determine if the column has more unique categories than max_categories
        num_unique = df[col].nunique(dropna=True)
        if num_unique > max_categories:
            value_counts = df[col].value_counts().nlargest(max_categories)
            print(f"# For variable '{col}', the top {max_categories} categories are displayed based on their value_counts:")
        else:
            value_counts = df[col].value_counts()
            print(f"# For variable '{col}', these are the categories available and their respective value_counts:")
        
        # Prepare value counts string
        value_counts_str = ', '.join([f"'{cat}': {count}" for cat, count in value_counts.items()])
        
        # Prepare list of categories as string
        categories_list_str = ', '.join([f"'{cat}'" for cat in value_counts.index])
        
        # Print the summaries as comments
        print(f"# {value_counts_str}")
        print("# Please choose which categories to include:")
        print(f"# [{categories_list_str}]")
        # Add the note about the behavior when both top_n and categories_to_keep are provided
        print("# Note: If both `top_n` and `categories_to_keep` are provided, `categories_to_keep` will be ignored.")
        # Add the note about the drop parameter
        print("# If drop = True rows will be dropped, otherwise labeled \"other\"")
        # Print the pre-filled filter_categories function call
        print(f"df = filter_categories(df, '{col}', drop=False, top_n=None, categories_to_keep=[{categories_list_str}])")
        print()  # Add an empty line for better readability
    
    # Handle IT columns separately
    IT_columns = ['IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5']
    IT_present = [col for col in IT_columns if col in df.columns]
    
    if IT_present:
        print(f"# Handling IT columns: {IT_present}")
        print("# Aggregating IT codes across IT_1 to IT_5 and listing the top categories:")
        
        # Concatenate all IT columns to compute global top_n
        combined_ITs = pd.concat([df[col] for col in IT_present], axis=0, ignore_index=True).dropna()
        IT_value_counts = combined_ITs.value_counts().nlargest(max_categories)
        
        # Prepare IT value counts string
        IT_value_counts_str = ', '.join([f"'{it}': {count}" for it, count in IT_value_counts.items()])
        
        # Prepare list of top ITs as string
        IT_list_str = ', '.join([f"'{it}'" for it in IT_value_counts.index])
        
        # Print IT categories summary as comments
        print(f"# {IT_value_counts_str}")
        print("# Please choose the number of top ITs to retain and include in the retain_top_n_ITs function call.")
        print(f"# Current top {max_categories} ITs:")
        print(f"# [{IT_list_str}]")
        
        # Print the pre-filled retain_top_n_ITs function call with a placeholder for top_n
        print(f"df = retain_top_n_ITs(df, top_n=10, IT_columns={IT_present}, other_label='Other')")
        print()  # Add an empty line for better readability
    else:
        print("# No IT columns found in the DataFrame.")
        
        

# Define Loading data function for the local drive
def load_data_local(file_name, file_path = data_path):
    """
    Loads a parquet or csv file from the local directory.

    Parameters:
    file_name (str): The name of the file to load.
    file_path (str): The path to the directory where the file is located.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # Full file path
    full_file_path = os.path.join(file_path, file_name)

    # Check file extension and load accordingly
    if file_name.endswith('.parquet'):
        print(f"Loading parquet file from local path: {full_file_path}")
        table = pq.read_table(full_file_path)
        df = table.to_pandas()  # Convert to pandas DataFrame
    elif file_name.endswith('.csv'):
        print(f"Loading csv file from local path: {full_file_path}")
        df = pd.read_csv(full_file_path)  # Read CSV into pandas DataFrame
    else:
        raise ValueError("Unsupported file format. Please provide a parquet or csv file.")

    return df

# dataset to be loaded must be in google-drive (shared: Project_CO2_DS/Data/EU Data)
# Add a shortcut by clicking to the 3 dots (file_name) to the left, click "Organise"
# click "Add shortcut", choose "My Drive" or another folder of your preferrence but
# also in "My Drive", click ADD...

def load_data_gdrive(file_name):
    """
    Ensures Google Drive is mounted, searches for a file by name across the
    entire Google Drive, and loads a parquet or csv file if found.

    Parameters:
    file_name (str): The name of the file to load.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame, or None if the file
    is not found.
    """
    # Function to check and mount Google Drive if not already mounted
    def check_and_mount_drive():
        """Checks if Google Drive is mounted in Colab, and mounts it if not."""
        drive_mount_path = '/content/drive'
        if not os.path.ismount(drive_mount_path):
            print("Mounting Google Drive...")
            # Import inside the condition when it's determined that mounting is needed
            from google.colab import drive
            drive.mount(drive_mount_path)
        else:
            print("Google Drive is already mounted.")

    # Function to search for the file in Google Drive
    def find_file_in_drive(file_name, start_path='/content/drive/My Drive'):
        """Search for a file by name in Google Drive starting from a specified path."""
        for dirpath, dirnames, filenames in os.walk(start_path):
            if file_name in filenames:
                return os.path.join(dirpath, file_name)
        return None

    # Check and mount Google Drive if not already mounted
    check_and_mount_drive()

    # Find the file in Google Drive
    file_path = find_file_in_drive(file_name)
    if not file_path:
        print("File not found.")
        return None

    # Check file extension and load accordingly
    if file_name.endswith('.parquet'):
        print(f"Loading parquet file: {file_path}")
        table = pq.read_table(file_path)
        df = table.to_pandas()  # Convert to pandas DataFrame
    elif file_name.endswith('.csv'):
        print(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)  # Read CSV into pandas DataFrame
    else:
        raise ValueError("Unsupported file format. Please provide a parquet or csv file.")

    return df

# Define saving data function
# Example usage:
# save_data(df, 'my_data.csv', '/path/to/save')

def save_data(df, file_name, file_path = data_path):
    """
    Saves a pandas DataFrame as a CSV file to the specified path.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame to save.
    file_name (str): The name of the CSV file to save (should end with .csv).
    file_path (str): The path to the directory where the file should be saved.

    Returns:
    None
    """
    # Full file path
    full_file_path = os.path.join(file_path, file_name)

    # Check if the file name ends with .csv
    if not file_name.endswith('.csv'):
        raise ValueError("File name should end with '.csv' extension.")

    # Save the DataFrame as CSV
    print(f"Saving DataFrame as a CSV file at: {full_file_path}")
    df.to_csv(full_file_path, index=False)

    print(f"CSV file saved successfully at: {full_file_path}")

    return full_file_path 


# Function to save file in google drive folder when working with colab 
def save_data_gdrive(df, file_name):
    """
    Ensures Google Drive is mounted and saves a DataFrame to Google Drive as a
    parquet or csv file, based on the file extension provided in file_name.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_name (str): The name of the file to save (with the .csv or .parquet extension).

    Returns:
    str: The path where the file was saved.
    """

    # Function to check and mount Google Drive if not already mounted
    def check_and_mount_drive():
        """Checks if Google Drive is mounted in Colab, and mounts it if not."""
        drive_mount_path = '/content/drive'
        if not os.path.ismount(drive_mount_path):
            print("Mounting Google Drive...")
            # Import inside the condition when it's determined that mounting is needed
            from google.colab import drive
            drive.mount(drive_mount_path)
        else:
            print("Google Drive is already mounted.")

    # Check and mount Google Drive if not already mounted
    check_and_mount_drive()

    # Define the saving directory in Google Drive (modify as needed)
    save_dir = '/content/drive/My Drive/Project_CO2_DS/Data/EU Data'

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Automatically detect the file format from the file_name extension
    if file_name.endswith('.parquet'):
        file_format = 'parquet'
    elif file_name.endswith('.csv'):
        file_format = 'csv'
    else:
        raise ValueError("Unsupported file format. Please provide a file name with '.parquet' or '.csv' extension.")

    # Full path to save the file
    file_path = os.path.join(save_dir, file_name)

    # Save the DataFrame based on the detected format
    if file_format == 'parquet':
        print(f"Saving DataFrame as a parquet file: {file_path}")
        df.to_parquet(file_path, index=False)
    elif file_format == 'csv':
        print(f"Saving DataFrame as a CSV file: {file_path}")
        df.to_csv(file_path, index=False)

    print(f"File saved at: {file_path}")
    return file_path


# Function to drop rows where all values in 'Ewltp (g/km)' are NaN
def drop_rows_without_target(df, target='Ewltp (g/km)'):
    """
    Drops rows where values in the specified target column are NaN.

    Parameters:
    df (DataFrame): The DataFrame to operate on.
    target (str): The column name to check for NaN values. Defaults to 'Ewltp (g/km)'.

    Returns:
    DataFrame: The modified DataFrame with rows dropped where the target column has NaN values.
    """
    df.dropna(subset=[target], inplace=True)
    return df

# Define function for columns dropping
def drop_irrelevant_columns(df, columns_to_drop):
    """
    Drops irrelevant columns from the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.
    columns_to_drop (list): List of columns to drop from the DataFrame.

    Returns:
    pd.DataFrame: The updated DataFrame with specified columns removed.
    """
    # Filter only the columns that exist in the DataFrame
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop the existing columns
    if existing_columns_to_drop:
        df.drop(existing_columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns: {existing_columns_to_drop}")
    else:
        print("No columns to drop were found in the DataFrame.")

    # Display the updated DataFrame
    print(df.columns)
    return df

# Function to identify electric cars and replace nans in column electric capacity and range with 0
def process_electric_car_data(df, replace_nan=False, make_electric_car_column=True):
    """
    Processes the electric car data by optionally creating a Non_Electric_Car column
    and filling NaN values in Electric range and z (Wh/km) columns.

    Parameters:
    df (pd.DataFrame): The DataFrame containing car data.
    replace_nan (bool): Whether to fill NaN values in Electric range and z columns. Default is True.
    make_electric_car_column (bool): Whether to create the Non_Electric_Car column. Default is True.

    Returns:
    pd.DataFrame: The updated DataFrame with the new column and/or filled NaN values.
    """
    if make_electric_car_column:
        # Create the Non_Electric_Car column: 1 if both Electric range and z are NaN, otherwise 0
        df['Non_Electric_Car'] = (df['Electric range (km)'].isna() & df['z (Wh/km)'].isna()).astype(str).astype('category')

    if replace_nan:
        # Fill NaN values in Electric range (km) and z (Wh/km) with 0
        df['Electric range (km)'].fillna(0, inplace=True)
        df['z (Wh/km)'].fillna(0, inplace=True)

    return df

# Function to select only certain years 
def filter_dataframe_by_year(df, year=2018):
    """
    Filters the DataFrame by removing rows with a 'Year' less than the specified year.

    Parameters:
    df (pd.DataFrame): The DataFrame to be filtered.
    year (int, optional): The year threshold for filtering. Default is 2018.

    Returns:
    pd.DataFrame: The filtered DataFrame with rows removed based on the year.
    """
    # Check if the column is named 'Year' or 'year'
    if 'Year' in df.columns:
        year_column = 'Year'
    elif 'year' in df.columns:
        year_column = 'year'
    else:
        raise ValueError("The DataFrame must have a 'Year' or 'year' column.")

    # Prompt the user for a year if year is not provided
    if year is None:
        year_input = input("Please enter a year (default is 2018): ")
        year = int(year_input) if year_input else 2018

    # Remove rows where the 'Year' or 'year' column is less than the specified year
    df = df[df[year_column] >= year]

    return df


# Function to replace outliers with the median in Gaussian-distributed columns using the IQR method
def replace_outliers_with_median(df, columns=None, IQR_distance_multiplier=1.5, apply_outlier_removal=True):
    """
    Replaces outliers in the specified Gaussian-distributed columns with the median value of those columns.
    Outliers are identified using the IQR method.

    Parameters:
    df (DataFrame): The DataFrame to operate on.
    columns (list): The columns to check for outliers. If None, defaults to 'gaussian_cols'.
    IQR_distance_multiplier (float): The multiplier for the IQR to define outliers. Default is 1.5.
    apply_outlier_removal (bool): If True, applies the outlier replacement. If False, returns the original DataFrame.

    Returns:
    DataFrame: The modified DataFrame with outliers replaced by median values (if applied).
    """
    
    # Check if outlier removal is enabled
    if not apply_outlier_removal:
        print("Outlier replacement not applied. Returning original DataFrame.")
        return df  # Return the original DataFrame without modifications

    # Use 'gaussian_cols' as default if columns is None
    if columns is None:
        try:
            columns = gaussian_cols  # Ensure 'gaussian_cols' is defined
        except NameError:
            raise ValueError("Default column list 'gaussian_cols' is not defined. Please specify the 'columns' parameter.")

    # Identify which columns are present in the DataFrame
    existing_columns = [col for col in columns if col in df.columns]
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns are not in the DataFrame and will be skipped: {missing_columns}")

    if not existing_columns:
        print("No valid columns found for outlier replacement. Returning original DataFrame.")
        return df

    # Calculate the first (Q1) and third (Q3) quartiles for existing columns
    Q1 = df[existing_columns].quantile(0.25)
    Q3 = df[existing_columns].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile Range (IQR)

    # Define the outlier condition based on IQR
    outlier_condition = ((df[existing_columns] < (Q1 - IQR_distance_multiplier * IQR)) |
                         (df[existing_columns] > (Q3 + IQR_distance_multiplier * IQR)))

    # Replace outliers with the median of each existing column
    for col in existing_columns:
        median_value = df[col].median()  # Get the median value of the column
        outliers = outlier_condition[col]
        num_outliers = outliers.sum()
        if num_outliers > 0:
            df.loc[outliers, col] = median_value  # Replace outliers with the median
            print(f"Replaced {num_outliers} outliers in column '{col}' with median value {median_value}.")

    print(f"DataFrame shape after replacing outliers in Gaussian columns: {df.shape}")

    return df


# Function to remove outliers from non-Gaussian distributed columns using the IQR method for individual rows
import numpy as np
import pandas as pd

def iqr_outlier_removal(df, columns=None, IQR_distance_multiplier=1.5, apply_outlier_removal=True):
    """
    Removes outliers in specified non-Gaussian distributed columns using the IQR method for individual rows.
    Outliers are capped to the lower and upper bounds defined by the IQR.

    Parameters:
    df (DataFrame): The DataFrame to operate on.
    columns (list): The columns to check for outliers. If None, defaults to 'non_gaussian_cols'.
    IQR_distance_multiplier (float): The multiplier for the IQR to define outliers. Default is 1.5.
    apply_outlier_removal (bool): If True, applies the outlier removal process. If False, returns the original DataFrame.

    Returns:
    DataFrame: The modified DataFrame with outliers capped at the lower and upper bounds (if applied).
    """
    # Check if outlier removal is enabled
    if not apply_outlier_removal:
        print("Outlier removal not applied. Returning original DataFrame.")
        return df.copy()  # Return a copy of the original DataFrame without modifications

    # Use 'non_gaussian_cols' as default if columns is None
    if columns is None:
        try:
            columns = non_gaussian_cols  # Ensure 'non_gaussian_cols' is defined
        except NameError:
            raise ValueError("Default column list 'non_gaussian_cols' is not defined. Please specify the 'columns' parameter.")

    # Identify which columns are present in the DataFrame
    existing_columns = [col for col in columns if col in df.columns]
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns are not in the DataFrame and will be skipped: {missing_columns}")

    if not existing_columns:
        print("No valid columns found for outlier removal. Returning original DataFrame.")
        return df.copy()

    # Create a copy of the DataFrame to store results
    outliers_removed = df.copy()

    # Loop through each existing column to remove outliers
    for col in existing_columns:
        # Calculate the first (Q1) and third (Q3) quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1  # Interquartile Range (IQR)

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - IQR_distance_multiplier * IQR
        upper_bound = Q3 + IQR_distance_multiplier * IQR

        # Identify outliers
        lower_outliers = df[col] < lower_bound
        upper_outliers = df[col] > upper_bound
        total_outliers = lower_outliers.sum() + upper_outliers.sum()

        # Cap the outliers at the lower and upper bounds
        outliers_removed[col] = np.where(df[col] < lower_bound, lower_bound,
                                         np.where(df[col] > upper_bound, upper_bound, df[col]))

        print(f"Capped {total_outliers} outliers in column '{col}' between {lower_bound} and {upper_bound}.")

    # Print the shape of the DataFrame after capping outliers
    print(f"DataFrame shape after capping outliers in non-Gaussian columns: {outliers_removed.shape}")

    return outliers_removed  # Return the modified DataFrame


# Generate Dictionary for handling NaNs
def generate_strategy_dict_code(df, default_strategy='drop', special_columns=None):
    """
    Generates a string of Python code that defines a strategy dictionary
    with all DataFrame columns set to the specified default strategy,
    excluding any special columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - default_strategy (str): The default strategy for handling NaNs. 
                              Valid strategies include 'drop', 'mean', 'median', 'mode', 'zero'.
                              Defaults to 'drop'.
    - special_columns (list or None): List of columns with special handling.
                                      These columns will be excluded from the strategy dict.

    Returns:
    - str: A formatted string representing the strategy dictionary with comments.
    """
    # Start the dictionary string
    dict_lines = ["strategy = {"]

    # Exclude special columns if provided
    if special_columns is None:
        special_columns = []
    
    for col in df.columns:
        if col in special_columns:
            continue  # Skip special columns
        # Use repr to handle any special characters in column names
        dict_lines.append(f"    {repr(col)}: '{default_strategy}',")
    
    # Close the dictionary
    dict_lines.append("}")
    
    # Add a comment about special columns
    if special_columns:
        comment = f"# Note: The following columns are handled specially and are excluded from this strategy dict: {special_columns}"
        dict_lines.insert(0, comment)
    
    # Join all lines into a single string with line breaks
    strategy_code = "\n".join(dict_lines)
    
    return strategy_code


# Handle NaNs
import pandas as pd

def handle_nans(df, strategy_dict):
    """
    Handle NaNs in a DataFrame based on specified strategies per column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - strategy_dict (dict): A dictionary where keys are column names and values are strategies.
        Strategies can be 'drop', 'mean', 'median', 'mode', or 'zero'.

    Returns:
    - pd.DataFrame: The DataFrame with NaNs handled as specified.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Validate input
    if not isinstance(strategy_dict, dict):
        raise ValueError("strategy_dict must be a dictionary with column names as keys and strategies as values.")
    
    # Separate drop strategies from fill strategies
    drop_cols = [col for col, strategy in strategy_dict.items() if strategy == 'drop']
    fill_cols = {col: strategy for col, strategy in strategy_dict.items() if strategy in ['mean', 'median', 'mode', 'zero']}
    
    # Check for invalid strategies
    valid_strategies = {'drop', 'mean', 'median', 'mode', 'zero'}
    invalid = [ (col, strat) for col, strat in strategy_dict.items() if strat not in valid_strategies]
    if invalid:
        invalid_str = ', '.join([f"{col}: {strat}" for col, strat in invalid])
        raise ValueError(f"Invalid strategies provided for columns: {invalid_str}. Valid strategies are 'drop', 'mean', 'median', 'mode', 'zero'.")
    
    # Check if all columns exist in the DataFrame
    missing_cols = [col for col in strategy_dict.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
    
    # Handle drop strategies
    if drop_cols:
        # Count NaNs in drop_cols before dropping
        na_counts_before = df[drop_cols].isna().sum()
        # Drop rows with NaNs in any of the drop_cols
        df = df.dropna(subset=drop_cols)
        # Count NaNs in drop_cols after dropping
        na_counts_after = df[drop_cols].isna().sum()
        # Calculate number of rows dropped per column
        rows_dropped = na_counts_before - na_counts_after
        for col in drop_cols:
            print(f"Column '{col}': Dropped {rows_dropped[col]} row(s) containing NaN(s).")
        print()  # Add a newline for readability
    
    # Handle fill strategies
    for col, strategy in fill_cols.items():
        # Count NaNs before filling
        na_before = df[col].isna().sum()
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with mean ({fill_value}).")
            else:
                raise TypeError(f"Cannot use 'mean' strategy on non-numeric column '{col}'.")
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with median ({fill_value}).")
            else:
                raise TypeError(f"Cannot use 'median' strategy on non-numeric column '{col}'.")
        elif strategy == 'mode':
            # Mode can return multiple values; take the first one
            mode_series = df[col].mode()
            if mode_series.empty:
                fill_value = None  # If mode is empty, set to None or some default
                print(f"Column '{col}': No mode found. NaNs remain as NaN.")
            else:
                fill_value = mode_series.iloc[0]
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with mode ({fill_value}).")
        elif strategy == 'zero':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = 0
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with zero (0).")
            else:
                raise TypeError(f"Cannot use 'zero' strategy on non-numeric column '{col}'.")
        
    return df




# Handle NaNs in Erwltp and Ernedc
import pandas as pd

def handle_nans_IT_related_columns(df, it_columns, target_columns, strategy='mean'):
    """
    Handles NaNs in target_columns based on the presence of values in it_columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - it_columns (list): List of columns (e.g., 'IT_1' to 'IT_5') to check for non-NaN values.
    - target_columns (list): Columns (e.g., 'Erwltp (g/km)', 'Enedc (g/km)') to handle based on the IT columns.
    - strategy (str): Strategy to replace NaNs in target_columns when IT columns have values.
                      Must be either 'mean' or 'median'.
    
    Returns:
    - df (pd.DataFrame): The DataFrame with special NaN handling applied.
    """
    # Validate strategy
    if strategy not in ['mean', 'median']:
        raise ValueError("Strategy must be either 'mean' or 'median'.")
    
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Initialize the replacement counts dictionary
    replacement_counts = {
        'IT_present': {col: 0 for col in target_columns},
        'IT_absent': {col: 0 for col in target_columns}
    }
    
    # Check if all IT columns exist
    missing_it_cols = [col for col in it_columns if col not in df.columns]
    if missing_it_cols:
        raise ValueError(f"The following IT columns are not in the DataFrame: {missing_it_cols}")
    
    # Check if target columns exist
    missing_target_cols = [col for col in target_columns if col not in df.columns]
    if missing_target_cols:
        raise ValueError(f"The following target columns are not in the DataFrame: {missing_target_cols}")
    
    # Create boolean masks
    it_any_not_nan = df[it_columns].notna().any(axis=1)
    it_all_nan = df[it_columns].isna().all(axis=1)
    
    for target_col in target_columns:
        # Calculate fill value based on the strategy
        if strategy == 'mean':
            fill_value = df.loc[it_any_not_nan, target_col].mean()
            strategy_applied = 'mean'
        elif strategy == 'median':
            fill_value = df.loc[it_any_not_nan, target_col].median()
            strategy_applied = 'median'
        
        # Identify NaNs to replace where IT is present
        mask_IT_present = it_any_not_nan & df[target_col].isna()
        count_IT_present = mask_IT_present.sum()
        df.loc[mask_IT_present, target_col] = df.loc[mask_IT_present, target_col].fillna(fill_value)
        replacement_counts['IT_present'][target_col] = count_IT_present
        
        # Identify NaNs to replace where IT is absent
        mask_IT_absent = it_all_nan & df[target_col].isna()
        count_IT_absent = mask_IT_absent.sum()
        df.loc[mask_IT_absent, target_col] = df.loc[mask_IT_absent, target_col].fillna(0)
        replacement_counts['IT_absent'][target_col] = count_IT_absent
        
        # Print replacement counts
        print(f"Column '{target_col}':")
        print(f"  - Replaced {count_IT_present} NaN(s) with {strategy_applied} ({fill_value}) where IT was present.")
        print(f"  - Replaced {count_IT_absent} NaN(s) with 0 where IT was absent.\n")
    
    return df



