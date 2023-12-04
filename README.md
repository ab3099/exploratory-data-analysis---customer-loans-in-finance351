# exploratory-data-analysis---customer-loans-in-finance351
# Table of Contents

1. [RDSDatabaseConnector](#rdsdatabaseconnector)
   - [How it Works](#how-it-works)
   - [Prerequisites](#prerequisites)
   - [Installation Instructions](#installation-instructions)
   - [Example Usage](#example-usage)

2. [Data Transformation and Analysis Toolkit](#data-transformation-and-analysis-toolkit)
   - [DataTransform Class](#datatransform-class)
   - [DataFrameInfo Class](#dataframeinfo-class)
   - [DataFrameTransform Class](#dataframetransform-class)
   - [Plotter Class](#plotter-class)
   
# RDSDatabaseConnector

This Python script demonstrates a basic implementation of an RDSDatabaseConnector class that connects to an RDS database, extracts data, saves it to a CSV file, and reads it back into a Pandas DataFrame.

## How it Works

The `RDSDatabaseConnector` class is designed to connect to an RDS database using provided credentials, extract data from a specific table, save it to a CSV file, and read the data back into a Pandas DataFrame. The script consists of the following components:

### `load_credentials` Function

The `load_credentials` function loads database credentials from a YAML file.

### `RDSDatabaseConnector` Class



#### Initialization

The class constructor (`__init__`) initializes the RDSDatabaseConnector instance with the provided credentials.

### Methods

#### `_init_engine` 

This private method initializes the SQLAlchemy engine for connecting to the RDS database.

#### `extract_def` 

The `extract_def` method extracts data from the RDS database table 'loan_payments' into a Pandas DataFrame.

#### `save_to_csv` 

The `save_to_csv` method saves a given DataFrame to a CSV file ('loan_payments_data.csv' by default).

#### `extract_csv` 

The `extract_csv` method reads data from a CSV file back into a Pandas DataFrame.
## Prerequisites

- Python (any version)
- Required Python packages: `yaml`, `sqlalchemy`, `pandas`

## Installation Instructions

1. Clone the repository:

    ```python
    git clone https://github.com/ab3099/exploratory-data-analysis---customer-loans-in-finance351.git
    ```

2. Navigate to the project directory:

    ```bash
    cd exploratory-data-analysis---customer-loans-in-finance351
    ```

3. Install the required packages:

    ```python
    pip install pandas
    pip install psycopg2-binary
    pip install SQLAlchemy 
    ```
4. Updating exploratory-data-analysis---customer-loans-in-finance351: 

    ```python
    git git add .
    git commit -m "add loan_payments_csv and db_utils.py"
    git push install pandas 
    ```
## Example Usage

```python
import yaml
from RDSDatabaseConnector import RDSDatabaseConnector

# Load credentials from YAML file
credentials_data = load_credentials("capabilities.yaml")

# Create RDSDatabaseConnector instance
rds_connector = RDSDatabaseConnector(credentials_data)

# Extract data from RDS database and save to CSV
df_from_rds = rds_connector.extract_def()
rds_connector.save_to_csv(df_from_rds)

# Extract data from CSV file
df_from_csv = rds_connector.extract_csv()

# Display the head of the DataFrame
print(df_from_csv.head())
```

## Data Transformation and Analysis Toolkit

To enhance your data exploration, this project includes three additional classes:

### 1. `DataTransform` Class

The `DataTransform` class is a subclass of `pd.DataFrame` designed to perform various transformations on a DataFrame. It includes methods to convert specific columns from float to int, handle NaN values, replace values in a column based on a dictionary, convert a 'term' column to an integer, and format date columns.

## Methods

#### `convert_float_to_int_with_mean(self, columns)`
Converts specific columns from float to int, handling NaN values and replacing them with the mean of each column.

### `convert_float_to_int_without_mean(self, columns)`
Converts specific columns from float to int, handling NaN values without using the mean. Drops rows with NaN values in the specified columns.

### `convert_term_to_int(self)`
Converts the 'term' column to int and renames it to 'term_months'.

### `date_format(self, date_columns)`
Formats date columns in the DataFrame to the 'YYYY-MM' format.

### `replace_values_in_column(self, column, replacements)`
Replaces values in a specific column based on the provided dictionary.

#### Example Usage

```python
# Create an instance of DataTransform
transformer = DataTransform(df_from_db)

# Columns to convert from float to int with mean
float_to_int_columns_with_mean = ['funded_amount', 'funded_amount_inv', 'annual_inc']

# Columns to convert from float to int without mean
float_to_int_columns_without_mean = ['mths_since_last_major_derog', 'collections_12_mths_ex_med', 'mths_since_last_record', 'mths_since_last_delinq']

# Columns to format as dates
date_columns_to_format = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

# Define replacements dictionary for 'loan_status'
loan_status_replacements = {
    'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid',
    'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
    'Late (31-120 days)': 'Late',
    'Late (16-30 days)': 'Late'
}

# Apply transformations
transformer.convert_float_to_int_with_mean(float_to_int_columns_with_mean)
transformer.convert_float_to_int_without_mean(float_to_int_columns_without_mean)
transformer.convert_term_to_int()
transformer.date_format(date_columns_to_format)
transformer.replace_values_in_column('loan_status', loan_status_replacements)

# Display the transformed DataFrame
print(transformer.head())
```

# DataTransform Class

The `DataTransform` class is a subclass of `pd.DataFrame` designed to perform various transformations on a DataFrame. It includes methods to convert specific columns from float to int, handle NaN values, replace values in a column based on a dictionary, convert a 'term' column to an integer, and format date columns.

## Methods

### `convert_float_to_int_with_mean(self, columns)`
Converts specific columns from float to int, handling NaN values and replacing them with the mean of each column.

### `convert_float_to_int_without_mean(self, columns)`
Converts specific columns from float to int, handling NaN values without using the mean. Drops rows with NaN values in the specified columns.

### `convert_term_to_int(self)`
Converts the 'term' column to int and renames it to 'term_months'.

### `date_format(self, date_columns)`
Formats date columns in the DataFrame to the 'YYYY-MM' format.

### `replace_values_in_column(self, column, replacements)`
Replaces values in a specific column based on the provided dictionary.

## Example Usage

```python
# Create an instance of DataTransform
transformer = DataTransform(df_from_db)

# Columns to convert from float to int with mean
float_to_int_columns_with_mean = ['funded_amount', 'funded_amount_inv', 'annual_inc']

# Columns to convert from float to int without mean
float_to_int_columns_without_mean = ['mths_since_last_major_derog', 'collections_12_mths_ex_med', 'mths_since_last_record', 'mths_since_last_delinq']

# Columns to format as dates
date_columns_to_format = ['issue_date', 'earliest_credit_line', 'last_payment_date', 'next_payment_date', 'last_credit_pull_date']

# Define replacements dictionary for 'loan_status'
loan_status_replacements = {
    'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid',
    'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
    'Late (31-120 days)': 'Late',
    'Late (16-30 days)': 'Late'
}

# Apply transformations
transformer.convert_float_to_int_with_mean(float_to_int_columns_with_mean)
transformer.convert_float_to_int_without_mean(float_to_int_columns_without_mean)
transformer.convert_term_to_int()
transformer.date_format(date_columns_to_format)
transformer.replace_values_in_column('loan_status', loan_status_replacements)

# Display the transformed DataFrame
print(transformer.head())
```
# DataFrameInfo Class

The `DataFrameInfo` class extends the functionality of the `DataTransform` class and provides additional methods to gather insights and information about a Pandas DataFrame.

### Methods

####  `describe_columns()`

Describe all columns in the DataFrame to check their data types.

####  `extract_statistical_values()`

Extract statistical values including median, standard deviation, and mean from the columns and the DataFrame.

###  `count_distinct_values()`

Count distinct values in categorical columns.

### `print_shape()`

Print out the shape of the DataFrame.

### `generate_null_counts()`

Generate a count/percentage count of NULL values in each column.

## Example Usage

```python

# Assume 'transformer' is an instance of DataTransform with your DataFrame already transformed

# Create an instance of DataFrameInfo
info = DataFrameInfo(transformer)

# Test describe_columns method
column_types = info.describe_columns()
print("Column Types:")
print(column_types)

# Test extract_statistical_values method
statistical_values = info.extract_statistical_values()
print("\nStatistical Values:")
print(statistical_values)

# Test count_distinct_values method
distinct_values = info.count_distinct_values()
print("\nDistinct Values:")
print(distinct_values)

# Test print_shape method
shape = info.print_shape()
print("\nDataFrame Shape:")
print(shape)
```

# DataFrameTransform: Advanced Data Transformation and Analysis Toolkit

The `DataFrameTransform` class is an extension of the `DataTransform` class, providing advanced functionalities for data preprocessing, handling null values, imputing missing data, identifying and handling outliers, and removing highly correlated columns.

## Methods

###  `check_null_before()`

Returns the count of null values in each column before any transformations.

###  `drop_columns_with_high_nulls(threshold=8)`

Drops columns with null values exceeding the specified threshold.

###  `impute(threshold=8)`

Imputes missing values based on the column type. Numeric columns are imputed using the median, and categorical columns are imputed using the mode.

###  `check_null()`

Returns the count of null values in each column after transformations.

###  `identify_skewed_columns(threshold=1)`

Identifies skewed columns based on a specified skewness threshold.

###  `apply_transformation(skewed_columns)`

Applies the most appropriate transformation (log or square root) to skewed columns to reduce skewness.

###  `handle_outliers_zscore(columns, threshold=3)`

Removes outliers in specified columns using the Z-score method. Outliers beyond the specified threshold are replaced with NaN, and rows with NaN values in specified columns are dropped.

###  `remove_highly_correlated_columns(correlation_threshold=0.8)`

Removes columns that are highly correlated based on the correlation threshold. This helps in reducing multicollinearity in the dataset.

## Example Usage

```python

# Assume 'transformer' is an instance of DataTransform with your DataFrame already transformed
transformer_1 = DataFrameTransform(transformer)

# Check null values before transformations
null_counts_before = transformer_1.check_null_before()
print("Null Values Before Transformations:")
print(null_counts_before)

# Drop columns with high null values
transformer_1 = transformer_1.drop_columns_with_high_nulls(threshold=8)

# Impute missing values
transformer_1 = transformer_1.impute(threshold=8)

# Check null values after transformations
null_counts_after = transformer_1.check_null()
print("\nNull Values After Transformations:")
print(null_counts_after)

# Identify and handle skewed columns
skewed_columns = transformer_1.identify_skewed_columns(threshold=1)
transformer_1 = transformer_1.apply_transformation(skewed_columns)

# Handle outliers using Z-score
outlier_columns = [['loan_amount', 'funded_amount', 'funded_amount_inv']]
transformer_1.handle_outliers_zscore(columns=outlier_columns, threshold=3)

# Remove highly correlated columns
transformer_1 = transformer_1.remove_highly_correlated_columns(correlation_threshold=0.8)

# Optionally, you may access the updated DataFrame
updated_df = transformer_1.df
```
# Plotter Class

The `Plotter` class provides a set of static and instance methods to visualize data analysis results through various plots. These methods are designed to facilitate exploratory data analysis and gain insights into the characteristics of the dataset.

## Methods

### `plot_significant_null_counts(null_counts, title, threshold=10, color='blue')`

Generate a bar chart to visualize NULL values for columns with significant nulls.

- **Parameters:**
  - `null_counts`: Pandas Series containing the NULL counts for each column.
  - `title`: Title for the plot.
  - `threshold`: Threshold to identify significant NULL counts (default is 10).
  - `color`: Color for the bars in the plot (default is 'blue').

### `plot_skewness(title='Skewness Analysis')`

Generate a bar chart to visualize skewness values for numeric columns.

- **Parameters:**
  - `title`: Title for the plot (default is 'Skewness Analysis').

### `plot_transformed_data(title='Transformed Data Visualization')`

Generate a bar chart to visualize transformed data for numeric columns.

- **Parameters:**
  - `title`: Title for the plot (default is 'Transformed Data Visualization').

### `plot_all_column_outliers(title='Boxplots for Outlier Detection')`

Generate a boxplot to visualize outliers in all columns simultaneously.

- **Parameters:**
  - `title`: Title for the plot (default is 'Boxplots for Outlier Detection').

### `plot_correlation_matrix(dataframe)`

Generate a heatmap to visualize the correlation matrix for numeric columns in a given dataframe.

- **Parameters:**
  - `dataframe`: Pandas DataFrame containing numeric columns.

