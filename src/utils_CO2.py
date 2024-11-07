# Importing necessary librarys
# import numpy as np
import pandas as pd
import numpy as np
import pyarrow.parquet as pq  # For working with parquet files
import re

import seaborn as sns
import matplotlib.pyplot as plt

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


################### Function(s) for Initial Processing of Raw Data #########################

def optimize_dtypes(df):
    """
    Optimize the data types of a pandas DataFrame to reduce memory usage.

    This function performs the following optimizations:
    - Converts all object type columns to the 'category' dtype.
    - Downcasts integer columns to the smallest possible integer subtype.
    - Downcasts float columns to the smallest possible float subtype.

    It also prints the total memory usage of the DataFrame before and after optimization,
    as well as the number of columns converted for each data type.

    Parameters:
    df (pandas.DataFrame): The DataFrame to optimize.

    Returns:
    pandas.DataFrame: The optimized DataFrame.
    """
    # Calculate total memory usage before optimization
    before_mem = df.memory_usage(deep=True).sum()
    print(f"Total memory usage before optimization: {before_mem / 1024**2:.2f} MB")
    
    # Convert all object type to category
    obj_cols = df.select_dtypes(include="object").columns
    num_obj = len(obj_cols)
    if num_obj > 0:
        df[obj_cols] = df[obj_cols].astype("category")
        print(f"Converted {num_obj} object column{'s' if num_obj > 1 else ''} to 'category' dtype.")
    else:
        print("No object columns to convert to 'category' dtype.")
    
    # Downcast integer columns in place
    int_cols = df.select_dtypes(include='int').columns
    num_int_before = len(int_cols)
    if num_int_before > 0:
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
        print(f"Downcasted {num_int_before} integer column{'s' if num_int_before > 1 else ''}.")
    else:
        print("No integer columns to downcast.")
    
    # Downcast float columns in place
    float_cols = df.select_dtypes(include='float').columns
    num_float_before = len(float_cols)
    if num_float_before > 0:
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
        print(f"Downcasted {num_float_before} float column{'s' if num_float_before > 1 else ''}.")
    else:
        print("No float columns to downcast.")
    
    # Calculate total memory usage after optimization
    after_mem = df.memory_usage(deep=True).sum()
    print(f"Total memory usage after optimization: {after_mem / 1024**2:.2f} MB")
    
    return df



def check_missing_value_markers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks and counts different representations of missing values in each column of a DataFrame.

    This function iterates through each column of the provided DataFrame and counts the occurrences
    of various missing value indicators, including `np.nan`, `pd.NA`, `-99`, and the string 
    representation `"-99"`. The counts are aggregated based on the data type of each column to 
    ensure accurate identification of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be analyzed for missing value representations.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the counts of each missing value representation for every column.
        The resulting DataFrame has the following columns:
            - 'Column': Name of the original DataFrame column.
            - 'np.nan': Count of `np.nan` values.
            - 'pd.NA': Count of `pd.NA` values.
            - '-99': Count of `-99` numeric values.
            - '"-99"': Count of `"-99"` string values.
    """
    # Initialize a list to store the counts for each column
    counts_list = []
    
    # Iterate through each column in the DataFrame
    for col in df.columns:
        data = df[col]
        
        # Initialize counts for different missing value representations
        nan_count = 0
        pd_na_count = 0
        minus_99_count = 0
        str_minus_99_count = 0

        # Calculate total missing values (np.nan and pd.NA)
        total_missing = data.isna().sum()
        
        # Handle numeric columns (floats and integers)
        if pd.api.types.is_numeric_dtype(data):
            # Count of np.nan values
            nan_count = total_missing
            
            # Count occurrences of -99
            minus_99_count = (data == -99).sum()
        
        # Handle categorical columns
        elif pd.api.types.is_categorical_dtype(data):
            # Count of pd.NA values
            pd_na_count = total_missing
            
            # Count occurrences of "-99" as a string
            str_minus_99_count = (data == "-99").sum()
        
        # Handle object (string) columns
        elif pd.api.types.is_object_dtype(data):
            # Recalculate total missing values including np.nan, pd.NA, and None
            total_missing = data.isna().sum()
            
            # Convert data to a NumPy array for element-wise comparison
            values = data.to_numpy()
    
            # Create masks to differentiate between np.nan and pd.NA
            nan_mask = pd.isna(values) & (values != pd.NA)
            pd_na_mask = pd.isna(values) & (values == pd.NA)
    
            # Count np.nan and pd.NA values
            nan_count = nan_mask.sum()
            pd_na_count = pd_na_mask.sum()
            
            # Count occurrences of -99 (as numeric) and "-99" (as string)
            minus_99_count = (data == -99).sum()
            str_minus_99_count = (data == "-99").sum()
        
        else:
            # For other data types, consider total missing as np.nan
            nan_count = total_missing
        
        # Append the counts for the current column to the list
        counts_list.append({
            'Column': col,
            'np.nan': nan_count,
            'pd.NA': pd_na_count,
            '-99': minus_99_count,
            '"-99"': str_minus_99_count
        })
    
    # Convert the list of counts to a DataFrame for better readability
    counts_df = pd.DataFrame(counts_list)
    
    return counts_df


################### Functions for Cleanup of Categorical valuse ##########################

        
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
        
        
        
def filter_categories(df, column, drop=False, top_n=None, categories_to_keep=None, other_label='Other', min_cat_percent=10):
    """
    Filter categories in a column based on top_n, an explicit list of categories to keep, and minimum category frequency.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical column.
    - column (str): The name of the categorical column.
    - drop (bool, optional):
        If True, drop rows with categories not in categories_to_keep or top_n,
        or that occur less than min_cat_percent% of all rows.
        If False, label them as 'Other'. Defaults to False.
    - top_n (int, optional): Number of top categories to keep based on frequency.
    - categories_to_keep (list, optional): List of categories to retain.
    - other_label (str, optional): Label for aggregated other categories. Defaults to 'Other'.
    - min_cat_percent (float, optional):
        Minimum percentage threshold for category frequency.
        Categories occurring less than this percentage of total rows will be considered 'Other' or dropped.
        For example, 10 corresponds to 10%. Defaults to 10.

    Returns:
    - pd.DataFrame: DataFrame with updated categorical column.

    Notes:
    - If categories_to_keep is provided, only categories in this list will be considered, even when top_n is specified.
    - The min_cat_percent criterion always applies.
    - All categories that have a value count of less than min_cat_percent% of total rows (including NaN rows) will be replaced by other_label or dropped, depending on the drop parameter.
    - If no categories meet the min_cat_percent threshold, all categories will be labeled as 'Other' or dropped.

    Raises:
    - ValueError: If neither top_n nor categories_to_keep is provided.
    """
    import pandas as pd

    initial_row_count = len(df)
    total_rows = initial_row_count

    category_counts = df[column].value_counts(dropna=False)
    min_count = (min_cat_percent / 100) * total_rows

    # Apply min_cat_percent threshold
    categories_meeting_threshold = category_counts[category_counts >= min_count].index.tolist()

    if categories_to_keep is not None:
        # Filter categories_to_keep based on min_cat_percent threshold
        categories_to_keep = [cat for cat in categories_to_keep if cat in categories_meeting_threshold]

        if top_n is not None:
            # Select top_n categories from categories_to_keep
            filtered_counts = category_counts[categories_to_keep]
            top_categories = filtered_counts.nlargest(top_n).index.tolist()
        else:
            top_categories = categories_to_keep

        if not top_categories:
            print(f"No categories in categories_to_keep meet the minimum frequency threshold of {min_cat_percent}%.")
            if drop:
                print(f"All rows will be dropped because no categories meet the threshold.")
                return df.iloc[0:0]  # Return empty DataFrame with same columns
            else:
                print(f"All categories will be labeled as '{other_label}'.")
    else:
        if top_n is not None:
            # Determine top_n categories that meet the min_cat_percent threshold
            top_categories = category_counts[category_counts >= min_count].nlargest(top_n).index.tolist()
        else:
            # Keep all categories that meet the threshold
            top_categories = categories_meeting_threshold

        if not top_categories:
            print(f"No categories meet the minimum frequency threshold of {min_cat_percent}%.")
            if drop:
                print(f"All rows will be dropped because no categories meet the threshold.")
                return df.iloc[0:0]  # Return empty DataFrame with same columns
            else:
                print(f"All categories will be labeled as '{other_label}'.")

    print(f"Categories to keep after applying thresholds: {top_categories}")

    if drop:
        # Drop rows where column is not in top_categories or is rare
        df = df[df[column].isin(top_categories) | df[column].isna()]
        rows_dropped = initial_row_count - len(df)
        print(f"Dropped {rows_dropped} rows where '{column}' is not in the specified categories or occur less than {min_cat_percent}% of total rows.")
    else:
        # Replace categories not in top_categories or that are rare with other_label
        if pd.api.types.is_categorical_dtype(df[column]):
            # Add 'Other' to categories if not present
            if other_label not in df[column].cat.categories:
                df[column] = df[column].cat.add_categories([other_label])
        df[column] = df[column].where(df[column].isin(top_categories) | df[column].isna(), other_label)
        num_replaced = (df[column] == other_label).sum()
        print(f"Replaced {num_replaced} values in '{column}' with '{other_label}' where not in specified categories or occur less than {min_cat_percent}% of total rows.")

    return df





# Retain top_n ITs 
def retain_top_n_ITs(df, top_n, IT_columns=['IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5'], other_label='Other', min_cat_percent=10):
    """
    Retain top_n ITs (innovative technology codes) that have a frequency greater than min_cat_percent and replace others with `other_label`.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the IT columns.
    - top_n (int): Number of top ITs to consider based on frequency.
    - IT_columns (list, optional): List of IT column names. Defaults to ['IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5'].
    - other_label (str, optional): Label for aggregated other ITs. Defaults to 'Other'.
    - min_cat_percent (float, optional): Minimum percentage threshold for IT frequency.
        ITs occurring less than this percentage of total IT entries will be replaced by other_label.
        For example, 10 corresponds to 10%. Defaults to 10.
    
    Returns:
    - pd.DataFrame: DataFrame with updated IT columns.
    """
    import pandas as pd

    print(f"Retaining top {top_n} ITs that occur more than {min_cat_percent}% and labeling others as '{other_label}'...")

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

    # Total number of IT entries (excluding NaNs)
    total_IT_entries = len(combined_ITs)

    # Get IT counts
    IT_counts = combined_ITs.value_counts()

    # Calculate minimum count based on min_cat_percent
    min_count = (min_cat_percent / 100) * total_IT_entries

    # Determine the top_n ITs that meet the min_cat_percent threshold
    top_ITs = IT_counts[IT_counts >= min_count].nlargest(top_n).index.tolist()

    # Handle the case where no IT meets the threshold
    if not top_ITs:
        print(f"No ITs meet the minimum frequency threshold of {min_cat_percent}%. All ITs will be labeled as '{other_label}'.")
        top_ITs = []

    else:
        print(f"Top ITs meeting the threshold: {top_ITs}")

    # Replace ITs not in top_ITs with other_label using vectorized operations
    for col in existing_IT_columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            # Add other_label to categories if not present
            if other_label not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories([other_label])

        # Replace ITs not in top_ITs with other_label
        df[col] = df[col].where(df[col].isin(top_ITs) | df[col].isna(), other_label)

        updated_unique = df[col].dropna().unique().tolist()
        print(f"Updated '{col}' categories: {updated_unique}")

    print("Replacement complete.\n")
    print("=======================\n")
    return df.copy()






# Generate categories lists and functions to choose
def generate_category_lists(df, max_categories=20, min_cat_percent=10, top_n=10):
    """
    Generates summaries of categories for specified columns and provides
    pre-filled `filter_categories` and `retain_top_n_ITs` function calls for easy integration.

    The output is formatted with comments and code snippets that can be
    directly copied into your codebase. You can adjust the `top_n` and `min_cat_percent`
    parameters, and manually delete or modify the categories in the `categories_to_keep` list.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the categorical columns.
    - max_categories (int, default=20): Maximum number of categories to display 
      based on value counts for columns with high cardinality.
    - min_cat_percent (float, default=10): Default value for `min_cat_percent` parameter in the printed function calls. 
      The minimum percentage threshold for category frequency.
      Categories occurring less than this percentage of total rows will be considered 'Other' or dropped.
    - top_n (int, default=10): Default value for `top_n` parameter in the printed function calls.
      Categories with the top_n highest value_counts will be selected.

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
        total_rows = len(df)
        value_counts = df[col].value_counts()
        
        if num_unique > max_categories:
            top_categories = value_counts.nlargest(max_categories)
            print(f"# For variable '{col}', the top {max_categories} categories are displayed based on their value_counts:")
        else:
            top_categories = value_counts
            print(f"# For variable '{col}', these are the categories available and their respective value_counts:")
        
        # Prepare value counts string
        value_counts_str = ', '.join([f"'{cat}': {count}" for cat, count in top_categories.items()])
        
        # Prepare list of categories as string
        categories_list_str = ', '.join([f"'{cat}'" for cat in top_categories.index])
        
        # Print the summaries as comments
        print(f"# {value_counts_str}")
        print("# Please choose which categories to include or adjust the 'top_n' and 'min_cat_percent' parameters.")
        print(f"# [{categories_list_str}]")
        # Add notes about the parameters
        print("# Note: If both `top_n` and `categories_to_keep` are provided, `categories_to_keep` will be ignored.")
        print("# If drop = True, rows will be dropped; otherwise, they will be labeled as 'Other'.")
        # Print the pre-filled filter_categories function call including top_n and min_cat_percent
        print(f"df = filter_categories(df, '{col}', drop=False, top_n={top_n}, categories_to_keep=[{categories_list_str}], other_label='Other', min_cat_percent={min_cat_percent})")
        print()  # Add an empty line for better readability
        
    # Handle IT columns separately
    IT_columns = ['IT_1', 'IT_2', 'IT_3', 'IT_4', 'IT_5']
    IT_present = [col for col in IT_columns if col in df.columns]
    
    if IT_present:
        print(f"# Handling IT columns: {IT_present}")
        print("# Aggregating IT codes across IT_1 to IT_5 and listing the top categories:")
        
        # Concatenate all IT columns to compute global counts
        combined_ITs = pd.concat([df[col] for col in IT_present], axis=0, ignore_index=True).dropna()
        total_IT_entries = len(combined_ITs)
        IT_value_counts = combined_ITs.value_counts()
        
        # Get top IT categories based on max_categories
        top_ITs = IT_value_counts.nlargest(max_categories)
        
        # Prepare IT value counts string
        IT_value_counts_str = ', '.join([f"'{it}': {count}" for it, count in top_ITs.items()])
        
        # Prepare list of top ITs as string
        IT_list_str = ', '.join([f"'{it}'" for it in top_ITs.index])
        
        # Print IT categories summary as comments
        print(f"# {IT_value_counts_str}")
        print("# Please choose the number of top ITs to retain and adjust the 'top_n' and 'min_cat_percent' parameters in the function call.")
        print(f"# Current top {max_categories} ITs:")
        print(f"# [{IT_list_str}]")
        
        # Print the pre-filled retain_top_n_ITs function call including top_n and min_cat_percent
        print(f"df = retain_top_n_ITs(df, top_n={top_n}, IT_columns={IT_present}, other_label='Other', min_cat_percent={min_cat_percent})")
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
        print(f"Loaded parquet file from local path.")
        table = pq.read_table(full_file_path)
        df = table.to_pandas()  # Convert to pandas DataFrame
    elif file_name.endswith('.csv'):
        print(f"Loaded csv file from local path.")
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

# Define function for dropping columns
def drop_irrelevant_columns(df, columns_dict):
    """
    Drops irrelevant columns from the given DataFrame based on categorized column lists.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.
    columns_dict (dict): Dictionary where keys are category names and values are lists of columns to drop.

    Returns:
    pd.DataFrame: The updated DataFrame with specified columns removed.
    """
    
    print("\n====== Dropping/Selecting Columns ======")
    
    for category, columns in columns_dict.items():
        # Filter only the columns that exist in the DataFrame
        existing_columns_to_drop = [col for col in columns if col in df.columns]
        
        # Drop the existing columns
        if existing_columns_to_drop:
            df = df.drop(existing_columns_to_drop, axis=1)
            print(f'Columns "{category}" have been dropped: {existing_columns_to_drop}')
        else:
            print(f'No columns to drop were found in the DataFrame for "{category}".')
    
    # Display the updated DataFrame columns
    print(f"Columns now present in the DataFrame: {df.columns.tolist()}")
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
    
    print(f"Dropped all years prior to {year}.")

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
    
    print("\n============== Outlier Handling =====================")
    print("\n====  Removing outliers from Gaussian columns  ======")
    
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


import pandas as pd

def replace_outliers_with_median_iterative(df, columns=None, IQR_distance_multiplier=1.5, apply_outlier_removal=True, max_iterations=10):
    """
    Iteratively replaces outliers in the specified Gaussian-distributed columns with the median value of those columns.
    Outliers are identified using the IQR method and the process continues until no outliers remain or the maximum number of iterations is reached.

    Parameters:
    df (DataFrame): The DataFrame to operate on.
    columns (list): The columns to check for outliers. If None, defaults to 'gaussian_cols'.
    IQR_distance_multiplier (float): The multiplier for the IQR to define outliers. Default is 1.5.
    apply_outlier_removal (bool): If True, applies the outlier replacement. If False, returns the original DataFrame.
    max_iterations (int): The maximum number of iterations to perform. Default is 10.

    Returns:
    DataFrame: The modified DataFrame with outliers replaced by median values (if applied).
    """
    
    print("\n============== Outlier Handling =====================")
    print("\n====  Removing outliers from Gaussian columns  ======")
    
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
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        # Calculate the first (Q1) and third (Q3) quartiles for existing columns
        Q1 = df[existing_columns].quantile(0.25)
        Q3 = df[existing_columns].quantile(0.75)
        IQR = Q3 - Q1  # Interquartile Range (IQR)
    
        # Define the outlier condition based on IQR
        lower_bound = Q1 - IQR_distance_multiplier * IQR
        upper_bound = Q3 + IQR_distance_multiplier * IQR
        outlier_condition = ((df[existing_columns] < lower_bound) | (df[existing_columns] > upper_bound))
    
        # Check if there are any outliers in the current iteration
        total_outliers = outlier_condition.sum().sum()
        print(f"Total outliers detected in this iteration: {total_outliers}")
        if total_outliers == 0:
            print(f"No outliers detected. Stopping after {iteration - 1} iterations.")
            break
    
        # Replace outliers with the median of each existing column
        for col in existing_columns:
            outliers = outlier_condition[col]
            num_outliers = outliers.sum()
            if num_outliers > 0:
                median_value = df[col].median()  # Get the median value of the column
                df.loc[outliers, col] = median_value  # Replace outliers with the median
                print(f"Replaced {num_outliers} outliers in column '{col}' with median value {median_value}.")
    
        # If reached maximum iterations, notify the user
        if iteration == max_iterations:
            print(f"Reached maximum iterations ({max_iterations}). Some outliers may still remain.")
    
    print(f"\nFinal DataFrame shape after replacing outliers: {df.shape}")
    return df



# Function to remove outliers from non-Gaussian distributed columns using the IQR method for individual rows
import numpy as np
import pandas as pd

def iqr_outlier_removal(df, columns=None, IQR_distance_multiplier=1.5, apply_outlier_removal=True):
    """
    Removes outliers in specified non-Gaussian distributed columns using the IQR method for individual rows.
    Outliers are capped to the lower and upper bounds defined by the IQR, modifying the original DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to operate on.
    columns (list): The columns to check for outliers. If None, defaults to 'non_gaussian_cols'.
    IQR_distance_multiplier (float): The multiplier for the IQR to define outliers. Default is 1.5.
    apply_outlier_removal (bool): If True, applies the outlier removal process. If False, returns the original DataFrame.

    Returns:
    DataFrame: The modified DataFrame with outliers capped at the lower and upper bounds (if applied).
    """
    
    print("\n===  Removing outliers from non-Gaussian columns  ===")
    
    # Check if outlier removal is enabled
    if not apply_outlier_removal:
        print("Outlier removal not applied. Returning original DataFrame.")
        return df  # Return the original DataFrame without modifications

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
        return df

    # Loop through each existing column to remove outliers
    for col in existing_columns:
        # Calculate the first (Q1) and third (Q3) quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1  # Interquartile Range (IQR)

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - IQR_distance_multiplier * IQR
        upper_bound = Q3 + IQR_distance_multiplier * IQR

        # Identify and cap outliers at the lower and upper bounds
        lower_outliers = df[col] < lower_bound
        upper_outliers = df[col] > upper_bound
        total_outliers = lower_outliers.sum() + upper_outliers.sum()

        df.loc[lower_outliers, col] = lower_bound
        df.loc[upper_outliers, col] = upper_bound

        print(f"Capped {total_outliers} outliers in column '{col}' between {lower_bound} and {upper_bound}.")

    # Print the shape of the DataFrame after capping outliers
    print(f"DataFrame shape after capping outliers in non-Gaussian columns: {df.shape}")

    return df  # Return the modified DataFrame



# Generate Dictionary for handling NaNs
def generate_strategy_dict_code(df, default_strategy='drop', special_columns=None):
    """
    Generates a string of Python code that defines a strategy dictionary
    with strategies for handling NaNs in each DataFrame column.
    Columns starting with "IT_" will automatically have the strategy 'leave_as_nan'.
    All other columns are set to the specified default strategy,
    excluding any special columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - default_strategy (str): The default strategy for handling NaNs. 
                              Valid strategies include 'drop', 'mean', 'median', 'mode', 'zero', 'leave_as_nan'.
                              Defaults to 'drop'.
    - special_columns (list or None): List of columns with special handling.
                                      These columns will be excluded from the strategy dict.

    Returns:
    - str: A formatted string representing the strategy dictionary with comments.
    """
    # Start the dictionary string
    dict_lines = ["nan_handling_strategy = {"]

    # Exclude special columns if provided
    if special_columns is None:
        special_columns = []
    
    for col in df.columns:
        if col in special_columns:
            continue  # Skip special columns
        if col.startswith('IT_'):
            strategy = 'leave_as_nan'
        else:
            strategy = default_strategy
        # Use repr to handle any special characters in column names
        dict_lines.append(f"    {repr(col)}: '{strategy}',")
    
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
        Strategies can be 'drop', 'mean', 'median', 'mode', 'zero', or 'leave_as_nan'.

    Returns:
    - pd.DataFrame: The DataFrame with NaNs handled as specified.
    """
    
    print("\n=========  Handle NaNs  ==============")
    
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Validate input
    if not isinstance(strategy_dict, dict):
        raise ValueError("strategy_dict must be a dictionary with column names as keys and strategies as values.")
    
    # Define valid strategies
    valid_strategies = {'drop', 'mean', 'median', 'mode', 'zero', 'leave_as_nan'}
    
    # Check for invalid strategies
    invalid = [ (col, strat) for col, strat in strategy_dict.items() if strat not in valid_strategies]
    if invalid:
        invalid_str = ', '.join([f"{col}: {strat}" for col, strat in invalid])
        raise ValueError(f"Invalid strategies provided for columns: {invalid_str}. Valid strategies are 'drop', 'mean', 'median', 'mode', 'zero', 'leave_as_nan'.")
    
    # Identify columns specified in strategy_dict that are not in the DataFrame
    missing_cols = [col for col in strategy_dict.keys() if col not in df.columns]
    if missing_cols:
        print(f"WARNING: The following columns specified in strategy_dict are not in the DataFrame and will be skipped: {missing_cols}")
    
    # Identify columns in the DataFrame that are not specified in strategy_dict
    unused_cols = [col for col in df.columns if col not in strategy_dict]
    if unused_cols:
        print(f"WARNING: The following columns are in the DataFrame but not specified in strategy_dict and will be left unchanged: {unused_cols}")
    
    # Proceed only with columns that are present in both
    valid_cols = [col for col in strategy_dict.keys() if col in df.columns]
    
    if not valid_cols:
        print("No valid columns to process. Exiting function.")
        print("=======================\n")
        return df  # Return the copied DataFrame unmodified
    
    # Handle drop strategies first
    drop_cols = [col for col in valid_cols if strategy_dict[col] == 'drop']
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
    
    # Handle other strategies
    for col in valid_cols:
        strategy = strategy_dict[col]
        if strategy == 'drop':
            continue  # Already handled drop strategies
        
        # Count NaNs before handling
        na_before = df[col].isna().sum()
        
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with mean ({fill_value}).")
            else:
                print(f"WARNING: Cannot use 'mean' strategy on non-numeric column '{col}'. NaNs left as is.")
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with median ({fill_value}).")
            else:
                print(f"WARNING: Cannot use 'median' strategy on non-numeric column '{col}'. NaNs left as is.")
        elif strategy == 'mode':
            # Mode can return multiple values; take the first one
            mode_series = df[col].mode()
            if mode_series.empty:
                print(f"Column '{col}': No mode found. NaNs remain as NaN.")
            else:
                fill_value = mode_series.iloc[0]
                df[col].fillna(fill_value, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with mode ({fill_value}).")
        elif strategy == 'zero':
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(0, inplace=True)
                print(f"Column '{col}': Replaced {na_before} NaN(s) with zero (0).")
            else:
                print(f"WARNING: Cannot use 'zero' strategy on non-numeric column '{col}'. NaNs left as is.")
        elif strategy == 'leave_as_nan':
            # No action needed; NaNs are left as is
            print(f"Column '{col}': Left {na_before} NaN(s) as is.")
        else:
            # This should not happen due to earlier validation
            print(f"WARNING: Unhandled strategy '{strategy}' for column '{col}'. No changes made.")
        
    print("=======================\n")
    
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
    
    print ("\n===  Handle NaNs in IT related Columns Erwltp and Ernedc ===")
    
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



def count_unique_it_codes(df):
    """
    Counts the unique IT codes across all columns that start with 'IT_'.
    Returns a pandas Series with IT codes as index and their counts as values.
    """
    print("Counting unique IT codes across all IT columns...")
    
    # Dynamically identify IT code columns that start with 'IT_'
    it_code_columns = [col for col in df.columns if col.startswith('IT_')]
    print(f"Identified IT code columns: {it_code_columns}")
    
    if not it_code_columns:
        print("No IT code columns found starting with 'IT_'. Returning empty Series.")
        return pd.Series(dtype=int)
    
    # Concatenate the IT code columns into a single Series
    combined_it_codes = pd.concat([df[col] for col in it_code_columns], axis=0, ignore_index=True)
    
    # Drop NaN values
    combined_it_codes = combined_it_codes.dropna()
    
    # Count the unique IT codes
    it_code_counts = combined_it_codes.value_counts()
    
    print(f"Total unique IT codes found: {it_code_counts.shape[0]}")
    print(f"Top 10 most common IT codes:\n{it_code_counts.head(10)}\n")
    
    return it_code_counts


def encode_top_its(df, n=0):
    """
    One-hot encodes the top 'n' most common IT codes across all columns that start with 'IT_'.
    If n=0, encodes all IT codes.
    Returns the modified DataFrame with one-hot encoded columns added and original 'IT_' columns removed.
    """
    
    print ("\n===  Encoding Categories of IT Columns (Innovative Technologies)  ===")
    
    
    if n == 0:
        print("Encoding all IT codes...")
    else:
        print(f"Identifying Top {n} IT codes for one-hot encoding...")
    
    # Step 1: Count the unique IT codes
    it_code_counts = count_unique_it_codes(df)
    
    # Step 2: Identify Top n Most Common IT codes, or all if n==0
    if n == 0:
        top_it_codes = it_code_counts.index.tolist()
        print(f"Total IT codes to encode: {len(top_it_codes)}\n")
    else:
        top_it_codes = it_code_counts.head(n).index.tolist()
        print(f"Top {n} IT codes: {top_it_codes}\n")
    
    # Dynamically identify IT code columns that start with 'IT_'
    it_code_columns = [col for col in df.columns if col.startswith('IT_')]
    print(f"Identified IT code columns: {it_code_columns}")
    
    if not it_code_columns:
        print("No IT code columns found starting with 'IT_'. Skipping one-hot encoding.")
        return df
    
    # Stack the IT code columns into a single Series
    it_combined = df[it_code_columns].stack().reset_index(level=1, drop=True)
    
    # Filter the Series to include only the selected IT codes
    it_codes_filtered = it_combined[it_combined.isin(top_it_codes)]
    print(f"Total IT code entries after stacking and filtering: {it_codes_filtered.shape[0]}")
    
    if it_codes_filtered.empty:
        print("No IT code entries to encode after filtering. Skipping one-hot encoding.")
        return df
    
    # One-hot encode the IT codes
    it_code_dummies = pd.get_dummies(it_codes_filtered, prefix='IT_code')
    
    # Ensure binary values by using 'any' instead of 'sum'
    it_code_dummies = it_code_dummies.groupby(it_code_dummies.index).any().astype(int)
    print(f"Shape of the one-hot encoded IT code DataFrame: {it_code_dummies.shape}")
    
    # Reindex the dummy DataFrame to match df
    it_code_dummies = it_code_dummies.reindex(df.index, fill_value=0)
    
    # Remove existing one-hot encoded IT code columns if they exist to avoid duplication
    existing_dummy_columns = [col for col in it_code_dummies.columns if col in df.columns]
    if existing_dummy_columns:
        print(f"Existing one-hot encoded IT code columns detected: {existing_dummy_columns}. Dropping them to reinitialize.")
        df.drop(columns=existing_dummy_columns, inplace=True)
    
    # Concatenate the one-hot encoded IT codes with df
    df = pd.concat([df, it_code_dummies], axis=1)
    print("One-hot encoding completed.\n")
    
    # Drop original 'IT_' columns
    df.drop(columns=it_code_columns, inplace=True)
    print(f"Original IT code columns {it_code_columns} removed after encoding.\n")
    print("=======================\n")
    
    return df





def encode_categorical_columns(df, exclude_prefix='IT_', drop_first=True):
    """
    One-hot encodes all unique values present in each categorical column in the DataFrame.
    Excludes columns starting with 'exclude_prefix' and avoids the dummy variable trap.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing categorical columns.
    - exclude_prefix (str): Prefix of column names to exclude from encoding.
    
    Returns:
    - pd.DataFrame: The DataFrame with one-hot encoded columns added and original categorical columns removed.
    """
    
    print ("\n======  Encoding Categorical Variables ===========")
    
    # Identify categorical columns to encode, excluding those starting with 'exclude_prefix'
    cat_columns = [
        col for col in df.select_dtypes(include=['category', 'object']).columns 
        if not col.startswith(exclude_prefix)
    ]
    print(f"Identified categorical columns to encode (excluding '{exclude_prefix}'): {cat_columns}")
    
    if not cat_columns:
        print("No categorical columns found for encoding. Skipping one-hot encoding.")
        return df

    # Adjust categories to include only the present values in each categorical column
    for col in cat_columns:
        df[col] = df[col].astype('category')  # Ensure the column is of type category
        df[col] = df[col].cat.remove_unused_categories()  # Remove categories not present in the column

    # One-hot encode with `drop_first=True` to avoid dummy variable trap
    df_encoded = pd.get_dummies(df, columns=cat_columns, prefix=cat_columns, drop_first=drop_first)
    print(f"One-hot encoding completed with dummy variable trap prevention for columns: {cat_columns}")
    
    # Print the number of encoded values (columns) and the encoded values per original categorical column
    for col in cat_columns:
        encoded_values = df[col].cat.categories.tolist()[1:]  # Exclude the first as baseline
        num_encoded_cols = len(encoded_values)
        print(f"\nColumn '{col}' encoded into {num_encoded_cols} values (baseline category dropped):")
        print(f"Encoded values: {encoded_values}")
    
    print("=======================\n")
    
    return df_encoded



# ================================================================
# ========== manage filenames ====================================
import os
import pandas as pd

# Define the base directories
BASE_PATH = os.path.abspath(os.path.join('..', 'data', 'Preprocessed'))
SRC_PATH = os.path.abspath(os.path.join('..', 'src'))
MAPPING_CSV = os.path.join(SRC_PATH, 'file_mapping.csv')

def create_or_append_file_mapping(filename_list, mapping_csv=MAPPING_CSV, base_path=BASE_PATH):
    """
    Creates a new file mapping DataFrame or appends new filenames to an existing mapping.

    Parameters:
    - filename_list (list): List of filename strings to add.
    - mapping_csv (str): Path to the CSV file storing the mapping.
    - base_path (str): Base directory where the files are located.
    """
    # Ensure the source directory exists
    os.makedirs(os.path.dirname(mapping_csv), exist_ok=True)
    
    # If the mapping CSV exists, load it; otherwise, create an empty DataFrame
    if os.path.exists(mapping_csv):
        file_df = pd.read_csv(mapping_csv)
        # Ensure the 'number' column is of integer type
        file_df['number'] = file_df['number'].astype(int)
        next_num = file_df['number'].max() + 1 if not file_df.empty else 1
    else:
        file_df = pd.DataFrame(columns=['number', 'filename'])
        next_num = 1

    # Append new filenames, avoiding duplicates
    new_entries = []
    existing_filenames = set(file_df['filename'])
    for filename in filename_list:
        if filename not in existing_filenames:
            new_entries.append({'number': next_num, 'filename': filename})
            next_num += 1
        else:
            print(f"Filename '{filename}' already exists. Skipping.")

    if new_entries:
        new_df = pd.DataFrame(new_entries)
        file_df = pd.concat([file_df, new_df], ignore_index=True)
        # Save the updated mapping back to CSV
        file_df.to_csv(mapping_csv, index=False)
        print(f"Mapping saved to file_mapping.csv")
    else:
        print("No new filenames to add.")

def load_file_mapping(mapping_csv=MAPPING_CSV):
    """
    Loads the file mapping DataFrame from a CSV file.

    Parameters:
    - mapping_csv (str): Path to the CSV file storing the mapping.

    Returns:
    - pandas.DataFrame: DataFrame mapping numbers to filenames.
    """
    if not os.path.exists(mapping_csv):
        raise FileNotFoundError(f"The mapping CSV '{mapping_csv}' does not exist.")

    file_df = pd.read_csv(mapping_csv)
    file_df['number'] = file_df['number'].astype(int)
    return file_df



def load_data_by_number(number, mapping_csv=MAPPING_CSV, base_path=BASE_PATH):
    """
    Loads the data file corresponding to the given number.

    Parameters:
    - number (int): The number corresponding to the desired file.
    - mapping_csv (str): Path to the CSV file storing the mapping.
    - base_path (str): Base directory where the files are located.

    Returns:
    - pandas.DataFrame or other: Loaded data.
    """
    # Load the mapping DataFrame
    file_df = load_file_mapping(mapping_csv)

    # Find the filename corresponding to the given number
    row = file_df[file_df['number'] == number]
    if row.empty:
        raise ValueError(f"No file found with number: {number}")

    filename = row.iloc[0]['filename']
    full_path = os.path.join(base_path, filename)

    # Check if the file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{full_path}' does not exist.")

    # Load the data using the existing load function
    data = load_data_local(full_path)
    print(f"Loaded file: '{filename}'")
    return data

# ================ end of manage filenames ===========================


# drop duplicates and calculate frequencies from #identical occurences

def drop_duplicates(df, subset=None, drop=True, preserve_weights=False):
    """
    Removes duplicate rows from the DataFrame. Optionally preserves and updates frequency counts.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - subset (list or None): Columns to consider for identifying duplicates. If None, all columns are considered except 'frequency'.
    - drop (bool): Whether to drop duplicates. Default is True.
    - preserve_weights (bool): Whether to preserve and sum frequency counts. Default is False.

    Returns:
    - pd.DataFrame: The DataFrame after processing duplicates.
    """
    
    # store initial count of rows
    initial_count = len(df)
    
    print ("\n=========  Dropping Duplicates  ==============")
    
    if preserve_weights:
        # Check if the frequency column exists
        freq_col_original = '#identical occurences'
        freq_col_new = 'frequency'
        if freq_col_original not in df.columns:
            raise ValueError(f"The DataFrame does not contain the '{freq_col_original}' column.")
        
        # Ensure the frequency column is integer before any operations
        try:
            df[freq_col_original] = df[freq_col_original].astype(int)
        except ValueError as e:
            raise ValueError(f"Cannot convert '{freq_col_original}' to integer. Ensure all values are integers.") from e
        
        # Calculate total frequency before any changes
        total_freq_before = df[freq_col_original].sum()
        
        # Rename the frequency column to 'frequency'
        df = df.rename(columns={freq_col_original: freq_col_new})
        
        # Define group columns based on the subset
        if subset is not None:
            group_cols = subset
        else:
            # If subset is None, consider all columns except 'frequency' as group columns
            group_cols = [col for col in df.columns if col != freq_col_new]
        
        # Debug: Print the group columns being used
        print(f"Grouping by columns: {group_cols}")
        
        # Group by the specified columns and sum the 'frequency'
        frequency_sum = df.groupby(group_cols, as_index=False)[freq_col_new].sum()
        
        # Debug: Check for any NaN or negative frequencies after summing
        if frequency_sum[freq_col_new].isnull().any():
            raise ValueError("NaN values found in summed frequencies.")
        if (frequency_sum[freq_col_new] < 0).any():
            raise ValueError("Negative values found in summed frequencies.")
        
        # Retain the first occurrence of each group for other columns
        # This step keeps the first occurrence's data and updates the frequency
        df_first = df.drop_duplicates(subset=group_cols, keep='first')
        
        # Drop the existing 'frequency' to avoid duplication before merging
        df_first = df_first.drop(columns=[freq_col_new], errors='ignore')
        
        # Merge the summed frequency back to the first occurrences
        df = pd.merge(df_first, frequency_sum, on=group_cols, how='left', validate='one_to_one')
        
        # Ensure the frequency column is integer after merging
        df[freq_col_new] = df[freq_col_new].astype(int)
        
        # Calculate total frequency after changes
        total_freq_after = df[freq_col_new].sum()
        
        # Print frequency totals
        print(f"Total frequency before drops: {total_freq_before}")
        print(f"Total frequency after drops: {total_freq_after}")
        
        # Verify that the frequencies match
        if total_freq_before != total_freq_after:
            print("Warning: Total frequencies before and after dropping duplicates do not match.")
            print(f"Difference: {total_freq_after - total_freq_before}")
        else:
            print("Total frequencies are consistent before and after dropping duplicates.")
        
    elif drop:
        # Remove duplicates as usual
        df = df.drop_duplicates(subset=subset, keep='first')
    
    final_count = len(df)
    dropped_count = initial_count - final_count
    
    print(f"Initial row count: {initial_count}")
    print(f"Final row count: {final_count}")
    print(f"Number of duplicates {'dropped' if drop else 'found'}: {dropped_count}")
    print("=======================\n")
    
    return df






# plot histograms per year even more sophisticated version (with lineplot)

def plot_normalized_histograms(
    df,
    attributes,
    row_var,
    col_wrap=5,
    height=3,
    aspect=1.5,
    bins=10,
    palette=None,
    norm=True,
    iqrfactor=1.5,
    ylimfactor=1,
    plot_type='histogram'
):
    """
    Plots multiple histograms on the same subplot with consistent bin widths and y-limits.
    If norm is True, data is normalized between the minimum and Q3 + iqrfactor * IQR.
    Zeros and outliers (values > Q3 + iqrfactor * IQR) are excluded in both cases.

    Parameters:
    - df: The DataFrame containing the data.
    - attributes: A list of column names (attributes) to plot.
    - row_var: The column used to facet the data (e.g., 'year').
    - col_wrap: Number of columns per row in the grid (default is 5).
    - height: Height of each subplot (default is 3).
    - aspect: Aspect ratio of each subplot (default is 1.5).
    - bins: Number of bins in the visible part of the distributions.
    - palette: Palette the attribute colors are chosen from.
    - norm: Boolean indicating whether to normalize the data or not.
    - iqrfactor: Attribute values (outliers) are cut at Q3 + IQR * iqrfactor (default = 1.5)
    - ylimfactor: Factor to multiply ylim of all graphs with (default = 1)
    - plot_type: 'histogram', 'line', or 'both' to choose the plot type (default='histogram')
    """
    # Create a copy of the dataframe to avoid modifying the original data
    df_copy = df.copy()
    
    # Prepare dictionaries to store processed data
    data_dict = {}
    data_min_list = []
    data_max_list = []
    
    # Exclude zeros and outliers, normalize if required
    for attribute in attributes:
        column = df_copy[attribute]
        # Replace zeros with NaN
        column = column.replace(0, np.nan)
        # Remove NaNs for processing
        non_nan_values = column.dropna()
        
        # Check if non_nan_values is empty
        if non_nan_values.empty:
            print(f"No valid data for attribute '{attribute}' after excluding zeros and NaNs.")
            continue
        
        # Calculate IQR (Interquartile Range)
        q1 = np.percentile(non_nan_values, 25)
        q3 = np.percentile(non_nan_values, 75)
        iqr = q3 - q1
        
        # Define upper bound for outliers
        upper_bound = q3 + iqrfactor * iqr
        
        # Exclude outliers
        column[column > upper_bound] = np.nan
        
        # Optionally normalize the data
        if norm:
            lower_bound = non_nan_values.min()
            # Avoid division by zero
            if upper_bound - lower_bound == 0:
                print(f"Cannot normalize attribute '{attribute}' because upper_bound equals lower_bound.")
                normalized_column = pd.Series(np.nan, index=column.index)
            else:
                normalized_column = (column - lower_bound) / (upper_bound - lower_bound)
            data_dict[attribute] = normalized_column
        else:
            # Use original data within bounds
            data_dict[attribute] = column
        
        # Collect min and max values
        data_min_list.append(data_dict[attribute].min())
        data_max_list.append(data_dict[attribute].max())
    
    # Check if data_dict is empty
    if not data_dict:
        print("No valid data available for plotting after preprocessing.")
        return
    
    # Compute common min and max for bin edges
    data_min = np.nanmin(data_min_list)
    data_max = np.nanmax(data_max_list)
    
    # Handle case where data_min == data_max
    if data_min == data_max:
        print("All data points have the same value. Adjusting data_min and data_max for binning.")
        data_min -= 0.5
        data_max += 0.5
    
    # Create common bin edges
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    
    # Get a color palette if none is passed
    if palette is None:
        palette = sns.color_palette("tab10", len(attributes))
           
    # Get unique values of row_var to create facets
    facet_values = df_copy[row_var].unique()
    n_facets = len(facet_values)
    n_cols = col_wrap
    n_rows = int(np.ceil(n_facets / n_cols))
    
    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * aspect * height, n_rows * height))
    axes = axes.flatten()
    
    # Initialize a variable to store maximum y-value
    max_count = 0
    
    # For each facet (e.g., year), compute the histogram counts and plot
    for i, (facet_value, ax) in enumerate(zip(facet_values, axes)):
        # Subset the data for this facet
        df_facet = df_copy[df_copy[row_var] == facet_value]
        
        # Flag to check if any data was plotted
        plotted = False
        
        # Plot histograms for each attribute
        for attribute, color in zip(attributes, palette):
            if attribute not in data_dict:
                continue  # Skip if no valid data for this attribute
            
            data = data_dict[attribute][df_facet.index].dropna()
            if data.empty:
                continue  # Skip if no data to plot
            
            # Calculate histogram counts
            counts, _ = np.histogram(data, bins=bin_edges)
            # Calculate bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Smooth counts using a simple moving average to reduce the impact of outliers
            counts_smooth = np.convolve(counts, np.ones(3)/3, mode='same')
            # Update max_count with the smoothed counts
            max_count = max(max_count, counts_smooth.max())
            
            # Plot according to plot_type
            if plot_type == 'histogram':
                # Plot histogram bars
                ax.hist(
                    data,
                    bins=bin_edges,
                    color=color,
                    edgecolor='black',
                    alpha=0.5,
                    label=attribute,
                    density=False
                )
            elif plot_type == 'line':
                # Plot line of counts
                ax.plot(
                    bin_centers,
                    counts_smooth,
                    color=color,
                    label=attribute
                )
            elif plot_type == 'both':
                # Plot histogram bars
                ax.hist(
                    data,
                    bins=bin_edges,
                    color=color,
                    edgecolor='black',
                    alpha=0.5,
                    label=attribute,
                    density=False
                )
                # Plot line of counts
                ax.plot(
                    bin_centers,
                    counts_smooth,
                    color=color,
                    label=f"{attribute} (line)"
                )
            else:
                raise ValueError("Invalid plot_type. Choose 'histogram', 'line', or 'both'.")
            
            plotted = True
        
        if plotted:
            ax.set_title(f"{row_var}: {facet_value}")
            ax.legend()
            # Set xlim
            ax.set_xlim(data_min, data_max)
        else:
            ax.set_visible(False)  # Hide the subplot if nothing was plotted
    
    # Set y-limits for all axes based on the maximum smoothed count
    for ax in axes:
        if ax.get_visible():
            ax.set_ylim(0, max_count * 1.1 * ylimfactor)
            ax.set_xlim(data_min, data_max)
            ax.set_xlabel('Normalized Value' if norm else 'Value')
    
    # Remove unused subplots if any
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    

import numpy as np
import pandas as pd
from scipy.stats import boxcox, boxcox_normmax

def expand_by_frequency(series, frequency):
    """
    Expands a series based on given frequency values, repeating each element
    according to its specified frequency.

    Parameters:
    - series: pandas.Series, the series to expand.
    - frequency: pandas.Series, the frequency (can be float) for each element in `series`.

    Returns:
    - A new Series with each value repeated according to its frequency (rounded to nearest integer).
    """
    # Remove NaNs from series only
    series = series.dropna()
    frequency = frequency.loc[series.index].round().astype(int)
    
    expanded_series = series.repeat(frequency).reset_index(drop=True)
    return expanded_series

def weighted_quantile(values, quantiles, sample_weight):
    """
    Computes weighted quantiles for the values in `values`.

    Parameters:
    - values: Array-like structure containing the data values.
    - quantiles: List of quantile values to compute (e.g., [0.25, 0.75]).
    - sample_weight: Array-like structure containing the frequency or weight (can be float) for each value in `values`.

    Returns:
    - A list of calculated quantile values in the same order as `quantiles`.
    """
    # Remove NaNs from values only
    values = np.array(values)
    sample_weight = np.array(sample_weight)
    valid_indices = np.isfinite(values)
    values, sample_weight = values[valid_indices], np.round(sample_weight[valid_indices]).astype(int)
    
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    
    weighted_cumsum = np.cumsum(sample_weight)
    total_weight = weighted_cumsum[-1]
    
    quantile_values = []
    for q in quantiles:
        threshold = q * total_weight
        quantile_values.append(values[weighted_cumsum >= threshold][0])
        
    return quantile_values

def weighted_skewness(series, frequency):
    """
    Calculates the weighted skewness of a series, reflecting distribution asymmetry.

    Parameters:
    - series: Array-like structure with values to calculate skewness.
    - frequency: Array-like structure with the frequency or weight (can be float) for each value.

    Returns:
    - The weighted skewness as a floating-point number.
    """
    # Remove NaNs from series only
    series = series.dropna()
    frequency = frequency.loc[series.index]
    
    mean = np.average(series, weights=frequency)
    variance = np.average((series - mean) ** 2, weights=frequency)
    skewness = np.average((series - mean) ** 3, weights=frequency) / (variance ** 1.5)
    return skewness

def weighted_kurtosis(series, frequency):
    """
    Calculates the weighted kurtosis of a series, indicating distribution tail heaviness.

    Parameters:
    - series: Array-like structure with values to calculate kurtosis.
    - frequency: Array-like structure with the frequency or weight (can be float) for each value.

    Returns:
    - The weighted kurtosis as a floating-point number, centered around 0 for a normal distribution.
    """
    # Remove NaNs from series only
    series = series.dropna()
    frequency = frequency.loc[series.index]
    
    mean = np.average(series, weights=frequency)
    variance = np.average((series - mean) ** 2, weights=frequency)
    kurtosis = np.average((series - mean) ** 4, weights=frequency) / (variance ** 2) - 3
    return kurtosis

def optimal_boxcox_lambda_weighted(series, frequency, coarse=False):
    """
    Calculates the optimal Box-Cox lambda for a weighted series by expanding 
    the series first based on frequency and then determining the lambda.

    Parameters:
    - series: pandas.Series, the series to apply Box-Cox transformation to.
    - frequency: pandas.Series, the frequency or weight (can be float) for each value in `series`.
    - coarse: bool, if True, performs a quick coarse estimation of lambda using `boxcox_normmax`.

    Returns:
    - The optimal lambda value for the Box-Cox transformation.
    """
    # Step 1: Remove invalid frequencies and series values
    frequency = frequency.loc[series.index]
    frequency = frequency.round().astype(int)
    frequency[frequency < 1] = 1  # Set minimum frequency to 1

    # Remove series values less than 1 (since they are invalid in your context)
    valid_series = series >= 1
    series = series[valid_series]
    frequency = frequency[valid_series]

    # Debug: Check for values less than 1 in series
    print("After filtering, values < 1 in series:", (series < 1).sum())

    # Step 2: Expand the series based on frequency
    expanded_series = expand_by_frequency(series, frequency)

    # Step 3: Remove NaNs, non-positive values, and infinite values
    expanded_series = expanded_series[np.isfinite(expanded_series)]  # Remove infinite values
    expanded_series = expanded_series[expanded_series >= 1]  # Keep values >= 1
    expanded_series = expanded_series.dropna()  # Drop NaNs

    # Convert expanded_series to float64 for higher precision
    expanded_series = expanded_series.astype(np.float64)

    # Debug: Check for values less than 1 in expanded series
    print("Expanded series statistics after filtering:")
    print(expanded_series.describe())
    print("Values < 1 in expanded series after filtering:", (expanded_series < 1).sum())

    # Step 4: Check if expanded_series is empty
    if expanded_series.empty:
        raise ValueError("The expanded series is empty after filtering. Cannot compute Box-Cox lambda.")

    # Step 5: Compute the optimal lambda for Box-Cox transformation
    if coarse:
        # Using the 'mle' method for a robust estimation
        optimal_lambda = boxcox_normmax(expanded_series, method='mle')
    else:
        # Fine-tuned estimation using full Box-Cox optimization
        _, optimal_lambda = boxcox(expanded_series)

    return optimal_lambda



import pandas as pd
import numpy as np
from scipy.stats import boxcox

def apply_boxcox_transformation_df(df, lambda_dict):
    """
    Applies the Box-Cox transformation to specified columns in a DataFrame using provided lambda values.
    Drops all rows where any of the specified columns have values less than 1.
    Replaces the original columns with their transformed versions.
    
    Parameters:
    - df: pandas.DataFrame
        The DataFrame containing the data.
    - lambda_dict: dict
        A dictionary where keys are column names and values are lambda parameters for the Box-Cox transformation.
    
    Returns:
    - pandas.DataFrame
        The DataFrame with transformed columns.
    
    Raises:
    - ValueError: If any specified columns are missing from the DataFrame or contain non-positive values after filtering.
    """
    # Ensure all specified columns exist in the DataFrame
    missing_cols = [col for col in lambda_dict if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
    
    # Create a mask to filter out rows where any specified column has values < 1
    valid_mask = df[list(lambda_dict.keys())].ge(1).all(axis=1)
    df_filtered = df.loc[valid_mask].copy()
    
    # Apply the Box-Cox transformation to each specified column
    for col, lambda_value in lambda_dict.items():
        series = df_filtered[col]
        
        # Check for strictly positive values
        if (series <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values after filtering.")
        
        # Identify valid indices (non-NaN and finite)
        valid_indices = series.index[series.notna() & np.isfinite(series)]
        valid_series = series.loc[valid_indices]
        
        # Apply Box-Cox transformation
        transformed_data = boxcox(valid_series, lmbda=lambda_value)
        
        # Replace the original column with transformed data
        df_filtered[col] = np.nan  # Initialize with NaN
        df_filtered.loc[valid_indices, col] = transformed_data
    
    return df_filtered



# Function for Gaussian columns: Replacing outliers with median, with weighted option
def replace_outliers_with_median_weighted(df, columns=None, IQR_distance_multiplier=1.5, apply_outlier_removal=True,
                                          use_weighted_quartiles=False, weight_column=None):
    """
    Replaces outliers in specified Gaussian-distributed columns with the median, with an option to use weighted quartiles.
    Outliers are identified using the IQR method.

    Parameters:
    - df (DataFrame): The DataFrame to operate on.
    - columns (list): The columns to check for outliers. If None, defaults to 'gaussian_cols'.
    - IQR_distance_multiplier (float): The multiplier for the IQR to define outliers. Default is 1.5.
    - apply_outlier_removal (bool): If True, applies the outlier replacement process. If False, returns the original DataFrame.
    - use_weighted_quartiles (bool): If True, uses weighted quartiles for IQR calculation. Default is False.
    - weight_column (str): The name of the column containing weights for weighted quartile calculation.

    Returns:
    - DataFrame: The modified DataFrame with outliers replaced by median values (if applied).
    """
    
    print("\n===  Replacing outliers in Gaussian columns with median  ===")
    
    if not apply_outlier_removal:
        print("Outlier replacement not applied. Returning original DataFrame.")
        return df

    if columns is None:
        try:
            columns = gaussian_cols
        except NameError:
            raise ValueError("Default column list 'gaussian_cols' is not defined. Please specify the 'columns' parameter.")

    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        print("No valid columns found for outlier replacement. Returning original DataFrame.")
        return df

    if use_weighted_quartiles and weight_column is None:
        raise ValueError("When 'use_weighted_quartiles' is True, 'weight_column' must be specified.")

    for col in existing_columns:
        if use_weighted_quartiles:
            Q1, Q3 = weighted_quantile(df[col], [0.25, 0.75], df[weight_column])
        else:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_distance_multiplier * IQR
        upper_bound = Q3 + IQR_distance_multiplier * IQR

        outlier_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
        num_outliers = outlier_condition.sum()

        # Calculate median value (weighted or unweighted)
        if use_weighted_quartiles:
            median_value = weighted_quantile(df[col], [0.5], df[weight_column])[0]
        else:
            median_value = df[col].median()

        df.loc[outlier_condition, col] = median_value
        print(f"Replaced {num_outliers} outliers in column '{col}' with median value {median_value}.")

    print(f"DataFrame shape after replacing outliers in Gaussian columns: {df.shape}")

    return df



# Function for non-Gaussian columns: Capping outliers with weighted option
def iqr_outlier_removal_weighted(df, columns=None, IQR_distance_multiplier=1.5, apply_outlier_removal=True, 
                                 use_weighted_quartiles=False, weight_column=None):
    """
    Removes outliers in specified non-Gaussian distributed columns using the IQR method for individual rows.
    Outliers are capped to the lower and upper bounds defined by the IQR, with an option to use weighted quartiles.

    Parameters:
    - df (DataFrame): The DataFrame to operate on.
    - columns (list): The columns to check for outliers. If None, defaults to 'non_gaussian_cols'.
    - IQR_distance_multiplier (float): The multiplier for the IQR to define outliers. Default is 1.5.
    - apply_outlier_removal (bool): If True, applies the outlier removal process. If False, returns the original DataFrame.
    - use_weighted_quartiles (bool): If True, uses weighted quartiles for IQR calculation. Default is False.
    - weight_column (str): The name of the column containing weights for weighted quartile calculation.

    Returns:
    - DataFrame: The modified DataFrame with outliers capped at the lower and upper bounds (if applied).
    """
    
    print("\n===  Removing outliers from non-Gaussian columns  ===")
    
    if not apply_outlier_removal:
        print("Outlier removal not applied. Returning original DataFrame.")
        return df

    if columns is None:
        try:
            columns = non_gaussian_cols
        except NameError:
            raise ValueError("Default column list 'non_gaussian_cols' is not defined. Please specify the 'columns' parameter.")

    existing_columns = [col for col in columns if col in df.columns]
    if not existing_columns:
        print("No valid columns found for outlier removal. Returning original DataFrame.")
        return df

    if use_weighted_quartiles and weight_column is None:
        raise ValueError("When 'use_weighted_quartiles' is True, 'weight_column' must be specified.")

    for col in existing_columns:
        if use_weighted_quartiles:
            Q1, Q3 = weighted_quantile(df[col], [0.25, 0.75], df[weight_column])
        else:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_distance_multiplier * IQR
        upper_bound = Q3 + IQR_distance_multiplier * IQR

        lower_outliers = df[col] < lower_bound
        upper_outliers = df[col] > upper_bound
        total_outliers = lower_outliers.sum() + upper_outliers.sum()

        df.loc[lower_outliers, col] = lower_bound
        df.loc[upper_outliers, col] = upper_bound

        print(f"Capped {total_outliers} outliers in column '{col}' between {lower_bound} and {upper_bound}.")

    print(f"DataFrame shape after capping outliers in non-Gaussian columns: {df.shape}")

    return df



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc

def plot_weighted_boxplots(df, weight_column, exclude_columns=None, IQR_distance_multiplier=1.5):
    """
    Plots boxplots of all float columns in the DataFrame, excluding specified columns,
    while taking weights into account by expanding the data.

    Parameters:
    - df: pandas.DataFrame, the DataFrame containing the data.
    - weight_column: str, the name of the column containing weights (e.g., 'frequency').
    - exclude_columns: list of str, columns to exclude from plotting. Default is None.
    - IQR_distance_multiplier: float, the multiplier for the IQR to define whisker length. Default is 1.5.
    """
    # Check that the weight_column exists in the DataFrame
    if weight_column not in df.columns:
        raise ValueError(f"The weight column '{weight_column}' does not exist in the DataFrame.")
    
    # If exclude_columns is None, set it to an empty list
    if exclude_columns is None:
        exclude_columns = []
    else:
        # Ensure exclude_columns is a list
        if not isinstance(exclude_columns, list):
            raise TypeError("`exclude_columns` should be a list of column names.")
    
    # Select all float columns
    float_columns = df.select_dtypes(include=['float64', 'float32']).columns.tolist()
    
    # Exclude specified columns
    columns_to_plot = [col for col in float_columns if col not in exclude_columns]
    
    # If no columns remain after exclusion, raise an error
    if not columns_to_plot:
        raise ValueError("No float columns left to plot after excluding the specified columns.")
    
    # Ensure the weight column is not in the list of columns to plot
    if weight_column in columns_to_plot:
        columns_to_plot.remove(weight_column)
    
    print(f"Columns to be plotted: {columns_to_plot}")
    
    for col in columns_to_plot:
        if col not in df.columns:
            print(f"Column '{col}' does not exist in the DataFrame. Skipping.")
            continue

        print(f"\nPlotting boxplot for column '{col}'")
        # Get the series and weight column
        series = df[col]
        frequency = df[weight_column]

        # Use expand_by_frequency to expand the series
        try:
            expanded_series = expand_by_frequency(series, frequency)
        except Exception as e:
            print(f"Error expanding column '{col}': {e}. Skipping.")
            continue

        # Prepare the data as a DataFrame for seaborn
        data = expanded_series.to_frame(name=col)

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[col], whis=IQR_distance_multiplier)
        plt.title(f"Boxplot of {col} (weighted)")
        plt.xlabel(col)
        plt.show()

        # Delete the expanded_series to free memory
        del expanded_series
        gc.collect()


from scipy.stats import boxcox

def analyze_extreme_values_with_frequency(df, column, lambda_value, frequency_column, iqr_multiplier=1.5):
    """
    Analyzes extreme values before and after BoxCox transformation using weighted quantiles.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The column name to analyze (e.g., 'ep (KW)' or 'ec (cm3)').
    - lambda_value (float): The lambda value for BoxCox transformation.
    - frequency_column (str): The name of the column containing frequencies.
    - iqr_multiplier (float): Multiplier for IQR to define extreme values. Default is 1.5.

    Returns:
    - None: Prints the pre- and post-BoxCox extreme values with frequencies.
    """
    
    # Step 1: Calculate weighted IQR bounds for the original column
    values = df[column]
    frequencies = df[frequency_column]

    # Calculate weighted Q1 and Q3
    Q1, Q3 = weighted_quantile(values, [0.25, 0.75], frequencies)
    IQR = Q3 - Q1

    # Define bounds for extreme values in the original column
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Identify extreme values in the original data
    extreme_values_original = values[(values < lower_bound) | (values > upper_bound)]
    expanded_extreme_original = expand_by_frequency(extreme_values_original, frequencies[extreme_values_original.index])

    # Step 2: Apply BoxCox transformation
    # Filter to valid positive values for BoxCox transformation
    valid_indices = values > 0
    transformed_values = pd.Series(np.nan, index=values.index)
    transformed_values.loc[valid_indices] = boxcox(values.loc[valid_indices], lmbda=lambda_value)

    # Step 3: Calculate weighted IQR bounds for the transformed column
    Q1_trans, Q3_trans = weighted_quantile(transformed_values[valid_indices], [0.25, 0.75], frequencies[valid_indices])
    IQR_trans = Q3_trans - Q1_trans

    # Define bounds for extreme values in the transformed column
    lower_bound_trans = Q1_trans - iqr_multiplier * IQR_trans
    upper_bound_trans = Q3_trans + iqr_multiplier * IQR_trans

    # Identify extreme values in the transformed data
    extreme_values_transformed = transformed_values[(transformed_values < lower_bound_trans) | (transformed_values > upper_bound_trans)]
    expanded_extreme_transformed = expand_by_frequency(extreme_values_transformed, frequencies[extreme_values_transformed.index])

    # Step 4: Print Results
    print(f"Extreme values for column '{column}' (before BoxCox):")
    print(expanded_extreme_original.value_counts().sort_index())
    
    print(f"\nWeighted IQR bounds for original data: [{lower_bound}, {upper_bound}]")
    print(f"Number of extreme values before BoxCox: {len(expanded_extreme_original)}\n")
    
    print(f"Extreme values for column 'boxcox_{column}' (after BoxCox):")
    print(expanded_extreme_transformed.value_counts().sort_index())
    
    print(f"\nWeighted IQR bounds for transformed data: [{lower_bound_trans}, {upper_bound_trans}]")
    print(f"Number of extreme values after BoxCox: {len(expanded_extreme_transformed)}")



