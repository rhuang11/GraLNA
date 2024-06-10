import pandas as pd

csv_file = pd.read_csv('~/GraLNA/RF-FR/results_rf_fr.csv')

with open(csv_file, 'r') as file:
    # Create a CSV reader
    reader = csv.DictReader(file)
    
    # Initialize empty lists to store the parsed data
    data = []
    
    # Iterate over each row in the CSV file
    for row in reader:
        # Parse the dictionary in the current row
        parsed_dict = {key: ast.literal_eval(value) for key, value in row.items()}
        
        # Append the parsed dictionary to the list
        data.append(parsed_dict)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)