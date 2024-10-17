import pandas as pd
import os

# Directory containing the CSV files
directory_path = 'files_combined'

# Output CSV file path
output_csv_path = 'spar_human.csv'

# Combine all CSV files into a single DataFrame
combined_df = pd.concat(
    [pd.read_csv(os.path.join(directory_path, f)) for f in os.listdir(directory_path) if f.endswith('.csv')],
    ignore_index=True
)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(output_csv_path, index=False)

print(f"Combined CSV files saved to {output_csv_path}")
df2 = pd.read_csv('spar_human.csv')
df2 = df2[df2['answer'].apply(lambda x: x is not None and (isinstance(ast.literal_eval(x), list) and len(ast.literal_eval(x)) != 0 if isinstance(x, str) else False))]
print(len(df2))
df2.to_csv('spar_human_filtered.csv', index=False)




# Now combined_df contains all rows from all CSV files

import json
import csv

# File paths
jsonl_file_path = 'path_to_your_file.jsonl'
csv_file_path = 'output_file.csv'

# Open the JSONL file and read lines
with open(jsonl_file_path, 'r') as file:
    # Create a CSV writer object
    csv_file = open(csv_file_path, 'w', newline='')
    writer = csv.writer(csv_file)
    
    # Reading the first line to extract keys for CSV column headers
    first_line = file.readline()
    first_record = json.loads(first_line)
    headers = first_record.keys()
    writer.writerow(headers)  # Write the headers to the CSV file
    
    # Write the first data line
    writer.writerow(first_record.values())
    
    # Iterate over each remaining line in the JSONL file
    for line in file:
        # Convert JSON string to a dictionary
        record = json.loads(line)
        # Write the values to the CSV file
        writer.writerow(record.values())
    
    # Close the CSV file
    csv_file.close()


import json

# File path
jsonl_file_path = 'path_to_your_file.jsonl'

# List to hold all records
all_records = []

# Open the JSONL file and read lines
with open(jsonl_file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Convert JSON string to a dictionary
        record = json.loads(line)
        # Append the dictionary to the list
        all_records.append(record)

# `all_records` now contains all data from the JSONL file as a list of dictionaries
print(all_records)
