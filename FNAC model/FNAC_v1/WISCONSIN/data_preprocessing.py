import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True, help='Name of the target variable')
parser.add_argument('--csv_filename', type=str, required=True, help='Name of the CSV file')
args = parser.parse_args()

# Load the data
data = pd.read_csv(args.csv_filename)

# Drop the 'id' column if it exists
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data[args.target])

# Save the train and test data to CSV files
train_data.to_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva/train_data.csv', index=False)
test_data.to_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva/test_data.csv', index=False)

print("Data preprocessing completed.")
print(f"Number of training samples: {len(train_data)}")
print(f"Number of testing samples: {len(test_data)}")
