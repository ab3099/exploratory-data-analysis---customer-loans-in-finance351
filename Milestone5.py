import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
transformed_df = pd.read_csv('transformed_data.csv')
relevant_columns = ['funded_amount_inv', 'recoveries', 'loan_amount', 'issue_date']
df_subset = transformed_df[relevant_columns].copy()



# Summarize currently what percentage of the loans are recovered against investor funding and the total amount funded
df_subset['recovered_percent'] = (df_subset['recoveries'] / df_subset['funded_amount_inv']) * 100
df_subset['funded_percent'] = (df_subset['funded_amount_inv'] / df_subset['loan_amount']) * 100
print(df_subset['recovered_percent'])
print(df_subset['funded_percent'])
# Visualize results on an appropriate graph
plt.figure(figsize=(12, 6))

# Plotting the percentage of loans recovered
plt.subplot(1, 2, 1)
sns.histplot(df_subset['recovered_percent'], bins=20, kde=True)
plt.title('Percentage of Loans Recovered')

# Plotting the percentage of total amount funded
plt.subplot(1, 2, 2)
sns.histplot(df_subset['funded_percent'], bins=20, kde=True)
plt.title('Percentage of Total Amount Funded')

plt.tight_layout()
plt.show()

# Calculate months since the last payment
transformed_df['months_since_last_payment'] = (pd.to_datetime('today') - transformed_df['last_payment_date']).dt.days // 30
print(pd.to_datetime('today'))

print(transformed_df['months_since_last_payment'])
print(transformed_df['months_since_last_payment'].describe())

transformed_df_6_months = transformed_df[transformed_df['months_since_last_payment'] <= 6]
print(transformed_df_6_months)
# Filter for loans with a last payment within the last 6 months
transformed_df_6_months = transformed_df[(transformed_df['months_since_last_payment'] >= 0) & (transformed_df['months_since_last_payment'] <= 6)]


# Calculate the percentage of the total amount recovered after 6 months
transformed_df_6_months['covered_6_months_percent'] = (transformed_df_6_months['recoveries'] / transformed_df_6_months['funded_amount_inv']) * 100
print(transformed_df_6_months['covered_6_months_percent'])
# Visualize the percentage of the total amount recovered after 6 months
plt.figure(figsize=(10, 6))
sns.histplot(transformed_df_6_months['recovered_6_months_percent'], bins=20, kde=True)
plt.title('Percentage of Total Amount Recovered After 6 Months')
plt.xlabel('Percentage of Total Amount Recovered')
plt.ylabel('Number of Loans')
plt.show()



# Task 2. Filter the DataFrame to include only charged-off loans
charged_off_loans = transformed_df[transformed_df['loan_status'] == 'Charged Off']

# Calculate the percentage of charged-off loans
charged_off_percentage = (charged_off_loans.shape[0] / transformed_df.shape[0]) * 100

# Calculate the total amount paid towards charged-off loans
total_amount_paid = charged_off_loans['funded_amount_inv'].sum()

# Print the results
print(f"Percentage of Charged Off Loans: {charged_off_percentage:.2f}%")
print(f"Total Amount Paid towards Charged Off Loans: ${total_amount_paid:.2f}")

#Calculate the loss in revenue these loans would have generated for the company if they had finished their term

charged_off_loans = transformed_df[transformed_df['loan_status'] == 'Charged Off'].copy()
print(charged_off_loans['term'].head())
charged_off_loans['remaining_term'] = charged_off_loans['term'].str.extract('(\d+)').astype(int)
print(charged_off_loans['remaining_term'].head())



# Calculate the projected loss for each charged-off loan
charged_off_loans['projected_loss'] = charged_off_loans['instalment'] * charged_off_loans['remaining_term']

print(charged_off_loans[['remaining_term', 'projected_loss']].head(30))

# Visualize the projected loss over the remaining term
plt.figure(figsize=(12, 6))
sns.barplot(x='remaining_term', y='projected_loss', data=charged_off_loans, ci=None)
plt.title('Projected Loss Over Remaining Term of Charged-Off Loans')
plt.xlabel('Remaining Term (Months)')
plt.ylabel('Projected Loss ($)')
plt.show()

#task 4
#percentage of customers behind on their loan payment

behind_on_payments_percentage = (len(transformed_df[transformed_df['loan_status'] == 'Late']) / len(transformed_df)) * 100
print(f"Percentage of customers currently behind on payments: {behind_on_payments_percentage:.2f}%")

# Total number and loss if all customers behind on payments were charged off:
behind_on_payments = transformed_df[transformed_df['loan_status'] == 'Late'].copy()
total_customers_behind_on_payments = len(behind_on_payments)
print(total_customers_behind_on_payments)
total_loss_if_charged_off = behind_on_payments['loan_amount'].sum()  # Assuming 'loan_amnt' is the loan amount column
print(f"Total number of customers behind on payments: {total_customers_behind_on_payments}")
print(f"Total loss if all customers behind on payments were charged off: ${total_loss_if_charged_off:.2f}")

#Projected loss if customers behind on payments finish the full loan term:

behind_on_payments['remaining_term'] = behind_on_payments['term'].str.extract('(\d+)').astype(int)
remaining_term_months = behind_on_payments['remaining_term']
projected_loss = (remaining_term_months * behind_on_payments['instalment']).sum()
print(f"Projected loss if customers behind on payments finish the full loan term: ${projected_loss:.2f}")

#Percentage of total expected revenue represented by late and charged-off customers:
total_expected_revenue = transformed_df['total_payment'].sum()
late_customers_expected_revenue = transformed_df[transformed_df['loan_status'] == 'Late']['total_payment'].sum()
charged_off_customers_expected_revenue = transformed_df[transformed_df['loan_status'] == 'Charged Off']['total_payment'].sum()

# Step 4: Calculate the percentage for each group
percentage_late_customers = (late_customers_expected_revenue / total_expected_revenue) * 100
percentage_charged_off_customers = (charged_off_customers_expected_revenue / total_expected_revenue) * 100

print(f"Percentage of total expected revenue for late customers: {percentage_late_customers:.2f}%")
print(f"Percentage of total expected revenue for Charged Off customers: {percentage_charged_off_customers:.2f}%")
#task 5 
# Step 1: Create a Subset of Relevant Columns
relevant_columns = ['grade', 'purpose', 'home_ownership', 'loan_status']  # Add more columns as needed
subset_df = transformed_df[relevant_columns]

# Step 2: Filter Data for Charged Off and Late Customers
charged_off_customers = subset_df[subset_df['loan_status'] == 'Charged Off']
late_customers = subset_df[subset_df['loan_status'] == 'Late']

# Step 3: Explore and Visualize
# Example: Countplot for Grade
plt.figure(figsize=(12, 6))
sns.countplot(x='grade', hue='loan_status', data=subset_df, order=sorted(subset_df['grade'].unique()))
plt.title('Loan Status vs. Grade')
plt.show()

# Example: Countplot for Purpose
plt.figure(figsize=(14, 6))
sns.countplot(x='purpose', hue='loan_status', data=subset_df, order=subset_df['purpose'].value_counts().index)
plt.title('Loan Status vs. Purpose')
plt.xticks(rotation=45, ha='right')
plt.show()

# Example: Countplot for Home Ownership
plt.figure(figsize=(8, 6))
sns.countplot(x='home_ownership', hue='loan_status', data=subset_df)
plt.title('Loan Status vs. Home Ownership')
plt.show()

# Step 3: Compare Indicators Between Charged Off and Potential Charged Off Loans
charged_off_loans = subset_df[subset_df['loan_status'] == 'Charged Off']
potential_charged_off_loans = subset_df[subset_df['loan_status'] == 'Late']
# Example: Compare Grade distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='grade', hue='loan_status', data=pd.concat([charged_off_loans, potential_charged_off_loans]),
              order=sorted(subset_df['grade'].unique()))
plt.title('Loan Status vs. Grade (Charged Off vs. Potential Charged Off)')
plt.show()
# Example: Compare Purpose distribution
plt.figure(figsize=(14, 6))
sns.countplot(x='purpose', hue='loan_status', data=pd.concat([charged_off_loans, potential_charged_off_loans]),
              order=subset_df['purpose'].value_counts().index)
plt.title('Loan Status vs. Purpose (Charged Off vs. Potential Charged Off)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Example: Compare Home Ownership distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='home_ownership', hue='loan_status', data=pd.concat([charged_off_loans, potential_charged_off_loans]))
plt.title('Loan Status vs. Home Ownership')
plt.show()
