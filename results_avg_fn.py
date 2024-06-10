import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv('~/GraLNA/RF-FR/results_rf_fr.csv')

# Filter the DataFrame where the 'topN' column is 0.01
filtered_df = df[df['topN'] == 0.01]

metrics_list = filtered_df['metrics'].apply(ast.literal_eval)

# Create a DataFrame from the list of dictionaries
metrics_df = pd.DataFrame(metrics_list.tolist())

# Calculate the mean for the selected metrics
averaged_metrics = metrics_df[['auc', 'sensitivity_topk', 'precision_topk', 'ndcg_at_k']].mean()

# Convert the result to a DataFrame with a single row for display purposes
averaged_df = averaged_metrics.to_frame().T

print(averaged_df)
