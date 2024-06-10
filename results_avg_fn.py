import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('~/GraLNA/RF-FR/results_rf_fr.csv')

# Filter the DataFrame where the 'topN' column is 0.01
filtered_df = df[df['topN'] == 0.01]

# Further filter the DataFrame to include only the years between 2003 and 2008
filtered_df = filtered_df[(filtered_df['Year'] >= 2003) & (filtered_df['Year'] <= 2008)]

# Select the columns of interest
columns_of_interest = ['auc', 'sensitivity_topk', 'precision_topk', 'ndcg_at_k']

# Calculate the mean for the selected columns
averaged_values = filtered_df[columns_of_interest].mean()

# Convert the result to a DataFrame with a single row for display purposes
averaged_df = averaged_values.to_frame().T

print(averaged_df)
