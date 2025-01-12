import os
import pandas as pd


acc_path = "ftp.ebi.ac.uk/pub/databases/scnmt_gastrulation/acc/feature_level"
dataframes = []

# Loop through all files in the directory
for filename in os.listdir(acc_path):
    print(f"processing filename: {filename}")
    if filename.endswith('.tsv.gz') or filename.endswith('.tsv'):  # Adjust for your file extensions
        file_path = os.path.join(acc_path, filename)
        # Read the file
        df = pd.read_csv(file_path, sep='\t')
        # Append to the list
        dataframes.append(df)

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
print(combined_df.head())

# Pivot the DataFrame to create a matrix
matrix = combined_df.pivot(index='sample', columns='id', values='rate')

# Fill missing values with 0 (or another value, depending on your use case)
matrix = matrix.fillna(0)

matrix.to_csv('chromatin_accessibility_matrix.csv')
