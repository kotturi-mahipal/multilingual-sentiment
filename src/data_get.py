import gzip
import json
import pandas as pd

# Load the dataset
input_file_path = "data/labeled_Electronics_5.json.gz"
data = []
with gzip.open(input_file_path, 'rt', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Create a pandas DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
output_file_path = "data/electronics_5.csv"
df.to_csv(output_file_path, index=False)

print(f"Saved the dataset to: {output_file_path}")
