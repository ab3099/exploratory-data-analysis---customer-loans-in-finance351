import yaml
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_credentials(file_name= "capabilities.yaml"):
    with open(file_name, mode= "r") as f: 
        credentials = yaml.safe_load(f)
        return credentials
credentials_data = load_credentials()
print(credentials_data)

class RDSDatabaseConnector: 
    def __init__(self, credentials):
        self.host= credentials.get('RDS_HOST', '')
        self.password = credentials.get('RDS_PASSWORD', '')
        self.user = credentials.get('RDS_USER', '')
        self.database = credentials.get('RDS_DATABASE', '')
        self.port = credentials.get('RDS_PORT', '')
        self._init_engine()

    def _init_engine(self):
        DATABASE_TYPE = 'postgresql'  
        DBAPI = 'psycopg2'
        self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
        self.engine_connect = self.engine.execution_options(isolation_level='AUTOCOMMIT').connect()
    
    def extract_def(self):
        loan_payments_data = pd.read_sql_table('loan_payments', self.engine)
        return loan_payments_data
    
    def save_to_csv(self, df, file_name_two='loan_payments_data.csv'):
        df.to_csv(file_name_two, index=False)
        
              
    def extract_csv(self, file_path ='loan_payments_data.csv'):
     df_pd = pd.read_csv(file_path)
     return df_pd
    
class DataTransform(pd.DataFrame): 
    
    def convert_float_to_int_with_mean(self, columns):
     """Convert specific columns from float to int, handling NaN values and replacing with mean."""
     for col in columns:
        if col in self.columns and self[col].dtype == 'float64':
            self[col].fillna(self[col].mean(), inplace=True)
            self[col] = self[col].astype('int64')
     return self

    def convert_float_to_int_without_mean(self, columns):
        """Convert specific columns from float to int, handling NaN values without using mean."""
        for col in columns:
            self = self.dropna(subset=[col]).copy()  # Make a copy before dropping NaN values
            self[col] = self[col].astype('int64')

    def convert_term_to_int(self):
        """Convert the 'term' column to int and rename to 'term_months'."""
        self['term_months'] = self['term'].str.replace(r'\D', '', regex=True).astype(int)
        self.drop(columns=['term'], inplace=True)
        return self
    
    def date_format(self, date_columns):
        """Format date columns in the DataFrame."""
        for col in date_columns:
            self[col] = pd.to_datetime(self[col], format='%b-%Y', errors='coerce')
            self[col] = self[col].dt.strftime('%Y-%m')
              # Replace non-date values with NaN
            self[col] = self[col].where(self[col].notna(), other=np.nan)

    def replace_values_in_column(self, column, replacements):
        """Replace values in a specific column based on the provided dictionary."""
        self[column] = self[column].replace(replacements) 

class DataFrameInfo(DataTransform):

    def describe_columns(self):
        """Describe all columns in the DataFrame to check their data types."""
        return self.dtypes

    def extract_statistical_values(self):
        """Extract statistical values: median, standard deviation, and mean from the columns and the DataFrame."""
        return self.describe()

    def count_distinct_values(self):
        """Count distinct values in categorical columns."""
        categorical_columns = self.select_dtypes(include=['object']).columns
        distinct_values = {col: self[col].nunique() for col in categorical_columns}
        return distinct_values

    def print_shape(self):
        """Print out the shape of the DataFrame."""
        return self.shape
    
    def generate_null_counts(self):
        """Generate a count/percentage count of NULL values in each column."""
        null_counts = self.isnull().sum()
        percentage_null = (null_counts / len(self.df)) * 100
        null_info = pd.DataFrame({'Null Counts': null_counts, 'Percentage Null': percentage_null})
        return null_info

    
class DataFrameTransform(DataTransform):
  
    def check_null_before(self):
        null_counts = self.isnull().sum() 
        return null_counts

     
    def drop_columns_with_high_nulls(self, threshold=8):
        """Drop columns with null values exceeding the specified threshold."""
        columns_to_drop = []
        for col in self.columns:
            null_counts = self[col].isnull().sum()
            percentage_null = (null_counts / len(self)) * 100
            if percentage_null > threshold:
             columns_to_drop.append(col)
        self = self.drop(columns=columns_to_drop)
        return self
    
    def impute(self,threshold=8):

        for col in self.columns: 
            null_counts = self[col].isnull().sum()
            percentage_null = (null_counts / len(self)) * 100
            if pd.api.types.is_numeric_dtype(self[col]):
                # Impute numeric columns with median
                if percentage_null <= threshold:
                    impute_value = self[col].median()
                    self[col].fillna(impute_value, inplace=True)
            else:
                self[col].fillna(self[col].mode().iloc[0], inplace=True)
        return self
    
    def check_null(self):
        null_counts = self.isnull().sum() 
        return null_counts
    
    def identify_skewed_columns(self, threshold=1):
    # Select only numeric columns
        numeric_columns = self.select_dtypes(include=['number']).columns
    # Calculate skewness for all columns
        skewness = self[numeric_columns].apply(lambda x: x.skew())
    
    # Identify skewed columns based on the threshold
        skewed_columns = skewness[abs(skewness) > threshold].index.tolist()
        return skewed_columns
    
    def apply_transformation(self, skewed_columns):
        """Apply the most popular best transformation to the skewed columns."""
    
        for col in skewed_columns:
            original_skewness = self[col].skew()

            # Apply log transformation
            transformed_col_log = np.log1p(self[col])
            reduction_log = abs(original_skewness) - abs(transformed_col_log.skew())

            # Apply square root transformation
            transformed_col_sqrt = np.sqrt(self[col])
            reduction_sqrt = abs(original_skewness) - abs(transformed_col_sqrt.skew())

            # Choose the best transformation
            transformations = {
                'log': (transformed_col_log, reduction_log),
                'sqrt': (transformed_col_sqrt, reduction_sqrt),
            }

            col_best_transformation, _ = max(transformations.items(), key=lambda x: x[1][1])

            # Apply the most popular best transformation to the column
            if col_best_transformation == 'log':
                self[col] = np.log1p(self[col])
            elif col_best_transformation == 'sqrt':
                self[col] = np.sqrt(self[col])

        return self
    
    def handle_outliers_zscore(self, columns, threshold=3):
        """
        Remove outliers in specified columns using Z-score method.

        Parameters:
        - columns: List of column names to remove outliers.
        - threshold: Z-score threshold beyond which data points are considered outliers.
        """
        z_scores = np.abs((self[columns] - self[columns].mean()) / self[columns].std())
        outliers = z_scores > threshold

        # Replace outliers with NaN
        self[columns] = np.where(outliers, np.nan, self[columns])

        # Drop rows with NaN values in specified columns
        self = self.dropna(subset=columns)

    def remove_highly_correlated_columns(self, correlation_threshold=0.8):
        # Calculate the correlation matrix
         # Select only numeric columns
        numeric_columns = self.select_dtypes(include=np.number)

        # Calculate the correlation matrix for numeric columns
        correlation_matrix = numeric_columns.corr()

        # Find highly correlated columns
        highly_correlated_columns = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_columns.append(colname)

        # Drop highly correlated columns
        self = self.drop(columns=highly_correlated_columns)

        # Optionally, you may return the updated DataFrame
        return self
    

class Plotter: 
    @staticmethod
    def plot_significant_null_counts(null_counts, title, threshold=10, color='blue'):
        """Generate a bar chart to visualize NULL values for columns with significant nulls."""
        significant_nulls = null_counts[null_counts > threshold]

        plt.figure(figsize=(10, 6))

        # Plot null counts for columns with significant nulls
        plt.bar(significant_nulls.index, significant_nulls, color=color, alpha=0.7)

        plt.xlabel('Columns')
        plt.ylabel('Null Counts')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.show()

   
    def plot_skewness(self, title='Skewness Analysis'):
        """Generate a bar chart to visualize skewness values."""
        # Extract numeric columns
        numeric_columns = self.select_dtypes(include=np.number).columns

        # Calculate skewness values for numeric columns
        skewness_values = self[numeric_columns].skew()

        plt.figure(figsize=(10, 6))

        # Plot skewness values
        plt.bar(skewness_values.index, skewness_values, color='purple', alpha=0.7)

        plt.xlabel('Columns')
        plt.ylabel('Skewness')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.show()


    def plot_transformed_data(self, title='Transformed Data Visualization'):
        """Generate a bar chart to visualize transformed data."""
        # Extract numeric columns
        numeric_columns = self.select_dtypes(include=np.number).columns

        plt.figure(figsize=(12, 6))


        for col in numeric_columns:
            plt.bar(col + '_transformed', self[col + '_transformed'].skew(), color='green', alpha=0.7)

        plt.xlabel('Columns')
        plt.ylabel('Transformed Values')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.show()

    
    def plot_all_column_outliers(self, title='Boxplots for Outlier Detection'):
        """Generate a boxplot to visualize outliers in all columns simultaneously."""
        plt.figure(figsize=(15, 8))

        ax = sns.boxplot(data=self)
        ax.set(xlabel='Columns', ylabel='Values')

        sns.boxplot(self)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.show()
    
    def plot_correlation_matrix(self, dataframe):
        numeric_columns = dataframe.select_dtypes(include=np.number)

    # Calculate the correlation matrix for numeric columns
        correlation_matrix = numeric_columns.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()

# Example Usage:
# Create an instance of RDSDatabaseConnector
rds_connector = RDSDatabaseConnector(credentials_data)

# Extract data from the database
df_from_db = rds_connector.extract_def()

# Save data to CSV
rds_connector.save_to_csv(df_from_db)

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
transformer.date_format(date_columns_to_format)
transformer.convert_term_to_int()

transformer.replace_values_in_column('loan_status', loan_status_replacements)


## Create an instance of DataFrameInfo
info = DataFrameInfo(transformer)
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

# Create an instance of DataFrameTransform
transformer_1 = DataFrameTransform(transformer)
null_counts_before= transformer_1.check_null_before()

plotter= Plotter()
plotter.plot_significant_null_counts(null_counts_before, 'Significant Null Counts Before Removal', threshold=12, color='red')
transformer_1.drop_columns_with_high_nulls(threshold=30)
transformer_1.impute(threshold=30)
transformer_1.check_null()
skewed_columns= transformer_1.identify_skewed_columns(threshold=1)
plotter.plot_skewness(transformer_1, title='Skewness Visualization')
transformer_1.apply_transformation(skewed_columns)
transformer_1.to_csv('new_dataframe.csv', index=False)
plotter.plot_transformed_data(transformer_1, title='Transformed Data Visualization')
plotter.plot_all_column_outliers(transformer_1, title='Boxplots for Outlier Detection After Transformation')


# Columns to handle outliers with Z-score method
outlier_columns_to_handle = ['loan_amount', 'funded_amount', 'funded_amount_inv']

# Handle outliers using Z-score
transformer_1.handle_outliers_zscore(columns=outlier_columns_to_handle, threshold=3)


# Plot boxplots for outlier detection after handling outliers
plotter.plot_all_column_outliers(transformer_1, title='Boxplots for Outlier Detection After Transformation')
plotter.plot_correlation_matrix(transformer_1)
transformer_1.remove_highly_correlated_columns(correlation_threshold=0.8)





