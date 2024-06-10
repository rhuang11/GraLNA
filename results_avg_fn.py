import pandas as pd

csv_file = pd.read_csv('~/GraLNA/RF-FR/results_rf_fr.csv')

df = pd.read_csv(csv_file)

# Initialize empty list to store parsed data
data = []

# Iterate over DataFrame rows
for _, row in df.iterrows():
    # Parse the dictionary in the current row
    parsed_dict = {key: ast.literal_eval(value) for key, value in row.items()}
    
    # Append the parsed dictionary to the list
    data.append(parsed_dict)

# Convert the list of dictionaries to a DataFrame
df_parsed = pd.DataFrame(data)

# Display the DataFrame
print(df_parsed)