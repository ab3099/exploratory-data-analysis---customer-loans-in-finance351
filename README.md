# exploratory-data-analysis---customer-loans-in-finance351
# RDSDatabaseConnector

This Python script demonstrates a basic implementation of an RDSDatabaseConnector class that connects to an RDS database, extracts data, saves it to a CSV file, and reads it back into a Pandas DataFrame.

## How it Works

The `RDSDatabaseConnector` class is designed to connect to an RDS database using provided credentials, extract data from a specific table, save it to a CSV file, and read the data back into a Pandas DataFrame. The script consists of the following components:

### `load_credentials` Function

The `load_credentials` function loads database credentials from a YAML file.

### `RDSDatabaseConnector` Class



#### Initialization

The class constructor (`__init__`) initializes the RDSDatabaseConnector instance with the provided credentials.

#### `_init_engine` Method

This private method initializes the SQLAlchemy engine for connecting to the RDS database.

#### `extract_def` Method

The `extract_def` method extracts data from the RDS database table 'loan_payments' into a Pandas DataFrame.

#### `save_to_csv` Method

The `save_to_csv` method saves a given DataFrame to a CSV file ('loan_payments_data.csv' by default).

#### `extract_csv` Method

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
