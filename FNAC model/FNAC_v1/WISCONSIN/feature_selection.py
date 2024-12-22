import argparse
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True, help='Name of the target variable')
args = parser.parse_args()

# Load the data
train_data = pd.read_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\train_data.csv')
X_train = train_data.drop(args.target, axis=1)
y_train = train_data[args.target]

# Drop the 'id' column if it exists
if 'id' in X_train.columns:
    X_train.drop(columns=['id'], inplace=True)

# Perform feature selection using chi2
selector = SelectKBest(score_func=chi2, k=9)
X_train_selected = selector.fit_transform(X_train, y_train)

# Get the selected feature names
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = np.array(X_train.columns)[selected_feature_indices]

print("Selected Features:")
print(selected_feature_names)

# Save the selected features to a CSV file
selected_features_info = pd.DataFrame({'Feature Name': selected_feature_names})
selected_features_info.to_csv(r'C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\selected_features_info.csv', index=False)

print("Feature selection completed.")
print(f"Number of selected features: {len(selected_feature_names)}")
