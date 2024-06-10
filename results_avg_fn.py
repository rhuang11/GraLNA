import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('~/GraLNA/RF-FR/results_rf_fr.csv')

# Filter the DataFrame where the 'topN' column is 0.01
filtered_df = df[df['topN'] == 0.01]

# Group by 'Year' and calculate the mean for each group
averaged_df = filtered_df.groupby('year_test').mean().reset_index()

print(averaged_df)
