import pandas as pd

df = pd.read_csv('~/GraLNA/RF-FR/results_rf_fr.csv')

data = df['metrics']

df2 = pd.DataFrame(data)

print(df2)