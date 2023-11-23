import yaml
from sqlalchemy import create_engine
import pandas as pd
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
    

rds_connector = RDSDatabaseConnector(credentials_data)
df_from_csv = rds_connector.extract_csv()

# Display the head of the DataFrame
print(df_from_csv.head())


print(df_pd.head()) 

rds_connector = RDSDatabaseConnector(credentials_data)
df = rds_connector.extract_def()
print(df.head())
rds_connector.save_to_csv(df)



      

