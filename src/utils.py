import numpy as np
import pandas as pd
import re

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
    - If both `top_n` and `categories_to_keep` are provided, `categories_to_keep` will be ignored, and only `top_n` will be used.
    
    Raises:
    - ValueError: If neither `top_n` nor `categories_to_keep` is provided.
    """
    if top_n is not None:
        # Ignore categories_to_keep if top_n is provided
        top_categories = df[column].value_counts().nlargest(top_n).index
        if drop:
            df = df[df[column].isin(top_categories)]
        else:
            if pd.api.types.is_categorical_dtype(df[column]):
                # Add 'Other' to categories if not present
                if other_label not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([other_label])
            df[column] = df[column].where(df[column].isin(top_categories), other_label)
    elif categories_to_keep is not None:
        if drop:
            df = df[df[column].isin(categories_to_keep)]
        else:
            if pd.api.types.is_categorical_dtype(df[column]):
                # Add 'Other' to categories if not present
                if other_label not in df[column].cat.categories:
                    df[column] = df[column].cat.add_categories([other_label])
            df[column] = df[column].where(df[column].isin(categories_to_keep), other_label)
    else:
        raise ValueError("Either top_n or categories_to_keep must be provided.")
    
    print(f"Column {column} has ben processed.")
    
    return df



# Retain top_n ITs 
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
    
    # Concatenate all IT columns to compute global top_n
    combined_ITs = pd.concat([df[col] for col in IT_columns if col in df.columns], axis=0, ignore_index=True).dropna()
    
    # Determine the top_n ITs
    top_n_ITs = combined_ITs.value_counts().nlargest(top_n).index.tolist()
    print(f"Top {top_n} ITs: {top_n_ITs}")
    
    # Replace ITs not in top_n with other_label using vectorized operations
    for col in IT_columns:
        if col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                # Add 'Other' to categories if not present
                if other_label not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([other_label])
            original_unique = df[col].dropna().unique().tolist()
            df[col] = df[col].where(df[col].isin(top_n_ITs), other_label)
            updated_unique = df[col].dropna().unique().tolist()
            print(f"Updated '{col}' categories: {updated_unique}")
    
    print("Replacement complete.\n")
    return df




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
    categorical_columns = ['Ct', 'Cr', 'Fm', 'Ft', 'Country', 'Mp', 'Mh']
    
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
