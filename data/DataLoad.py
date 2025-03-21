# ./run_synthea --exporter.baseDirectory="./MA/" Massachusetts -s 12345 -p 500 --exporter.csv.export=true
import pandas as pd
import os

# Define the folder path
folder_path = r'F:\GitHub\Weaver\synthea\MA\csv'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Load each CSV file into a DataFrame and store them in a dictionary
data_frames = {}
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    data_frames[file] = pd.read_csv(file_path)

# Example: Access a specific DataFrame
# df = data_frames['example.csv']